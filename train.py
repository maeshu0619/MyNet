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

from models.network import Network
import models.network as network_module
from models.utils.loss.loss import Loss
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
from models.utils.training.compression_primary_loss import _compression_primary_support_balance
from models.utils.training.full_context_subtree_loss import build_full_context_subtree_delta_loss
from models.utils.training.case_debug import *
from models.utils.training.metric_csv import *
from models.utils.training.metric_columns import LOSS_GRAD_PROBE_COLUMNS, PHASE7_EVAL_SUMMARY_COLUMNS
from models.utils.training.actual_codec_status import *
from models.utils.training.metric_rows import *
from models.utils.training.lr_control import apply_optimizer_lr_floor, step_scheduler_with_floor, optimizer_lrs_safe
from models.utils.training.episode_metrics import *
from models.utils.training.checkpoint_metrics import *
from models.utils.training.actual_compression_guard import apply_actual_compression_guard
from models.utils.training.for_better_logging import *
from models.utils.training.train_flow import * # train loopのStage固定、Subtree入力、1Subtree選択、圧縮目的合成、Epoch窓選択を使う
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
    # 8iは4シーケンスのうち先頭3つだけを学習に使う。
    if str(getattr(args, "dataname", "")).strip().lower() == "8i":
        return list(seq_dirs[:3])
    return list(seq_dirs)

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
):
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
    fast_diag_drop_set, fast_diag_global = _sparsepcgc_fast_diag_global_drop_set(full_coords, args)
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

    scored = []
    with torch.no_grad():
        for key in pool_keys:
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
            if isinstance(detail, dict):
                detail = dict(detail)
                detail["fast_diag_local_count"] = int(fast_local_count)
                detail["fast_diag_local_ratio"] = float(fast_local_ratio)
                detail["fast_diag_score"] = float(fast_score)
                detail["fast_diag_global_drop_count"] = int(fast_diag_global.get("global_drop_count", 0) or 0)
                detail["fast_diag_global_drop_ratio"] = float(fast_diag_global.get("global_drop_ratio", 0.0) or 0.0)
            scored.append((float(score), int(key), detail))

    if not scored:
        return None, {"enabled": True, "reason": "no_scored_groups", "pool": len(pool_keys)}

    scored.sort(key=lambda item: item[0], reverse=True)
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


def _sparsepcgc_splice_subtree_coords_into_full_cloud(full_coords_b3n, subtree_coords_n3, candidate_coords_n3):
    if not torch.is_tensor(full_coords_b3n) or not torch.is_tensor(subtree_coords_n3) or not torch.is_tensor(candidate_coords_n3):
        return None
    if full_coords_b3n.ndim == 2:
        full_coords_b3n = (
            full_coords_b3n.transpose(0, 1).contiguous().unsqueeze(0)
            if full_coords_b3n.shape[-1] == 3
            else full_coords_b3n.unsqueeze(0)
        )
    if full_coords_b3n.ndim != 3 or full_coords_b3n.shape[1] != 3 or full_coords_b3n.shape[0] != 1:
        return None
    device = candidate_coords_n3.device
    full_coords = torch.unique(
        full_coords_b3n[0].transpose(0, 1).contiguous().to(device=device, dtype=torch.long),
        dim=0,
        sorted=True,
    )
    subtree_coords = torch.unique(subtree_coords_n3.to(device=device, dtype=torch.long), dim=0, sorted=True)
    candidate_coords = torch.unique(candidate_coords_n3.to(device=device, dtype=torch.long), dim=0, sorted=True)
    if full_coords.numel() <= 0 or subtree_coords.numel() <= 0 or candidate_coords.numel() <= 0:
        return None

    remove_keys = {tuple(int(v) for v in row) for row in subtree_coords.detach().cpu().tolist()}
    keep_mask_cpu = [tuple(int(v) for v in row) not in remove_keys for row in full_coords.detach().cpu().tolist()]
    keep_mask = torch.as_tensor(keep_mask_cpu, device=device, dtype=torch.bool)
    spliced = torch.cat([full_coords[keep_mask], candidate_coords], dim=0)
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
        spliced_base = _sparsepcgc_splice_subtree_coords_into_full_cloud(
            full_octree_context.get("full_global_voxel_coords", None),
            unique_coords,
            unique_coords,
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
            if full_cloud_macro_eval_limit > 0 and torch.is_tensor(full_eval_coords):
                full_macro_generate_start = time.time()
                full_macro_candidates = _sparsepcgc_actual_oracle_full_cloud_macro_prune_candidates(
                    full_eval_coords,
                    args,
                    full_cloud_macro_max,
                    teacher_coords=unique_coords,
                )
                debug["full_cloud_macro_generate_time"] = float(time.time() - full_macro_generate_start)
                debug["generated_candidate_count"] = int(debug.get("generated_candidate_count", 0)) + int(len(full_macro_candidates))
                min_improve = max(float(getattr(args, "sparsepcgc_actual_oracle_min_improve_percent", 0.0)), 0.0)
                for full_macro_item in full_macro_candidates:
                    if _actual_budget_exhausted(tested) or full_cloud_macro_tested >= full_cloud_macro_eval_limit:
                        break
                    candidate_coords = full_macro_item.get("candidate_coords", None)
                    drop_coords = full_macro_item.get("drop_coords", None)
                    if not torch.is_tensor(candidate_coords) or not torch.is_tensor(drop_coords):
                        continue
                    local_map_start = time.time()
                    if str(full_macro_item.get("op", "")) == "full_cloud_subtree_prune":
                        drop_block_coords = full_macro_item.get("drop_block_coords", None)
                        block_size = max(int(full_macro_item.get("block_size", 32)), 2)
                        if not torch.is_tensor(drop_block_coords):
                            continue
                        drop_block_key_set = {
                            tuple(int(v) for v in row)
                            for row in drop_block_coords.detach().cpu().tolist()
                        }
                        local_blocks = torch.div(unique_coords, block_size, rounding_mode="floor")
                        local_unique_indices = [
                            int(idx)
                            for idx, row in enumerate(local_blocks.detach().cpu().tolist())
                            if tuple(int(v) for v in row) in drop_block_key_set
                        ]
                        drop_key_set = None
                    else:
                        drop_key_set = {tuple(int(v) for v in row) for row in drop_coords.detach().cpu().tolist()}
                        local_unique_indices = [
                            int(idx)
                            for idx, row in enumerate(unique_coords.detach().cpu().tolist())
                            if tuple(int(v) for v in row) in drop_key_set
                        ]
                    debug["full_cloud_macro_local_map_time"] = float(
                        debug.get("full_cloud_macro_local_map_time", 0.0)
                        + (time.time() - local_map_start)
                    )
                    restore_start = time.time()
                    candidate_xyz = _restore_codec_xyz_from_global_voxels(
                        args,
                        candidate_coords.transpose(0, 1).contiguous().unsqueeze(0),
                        full_octree_context if isinstance(full_octree_context, dict) else subtree_tree,
                        subtree_xyz,
                    )
                    if candidate_xyz is None or candidate_xyz.shape[-1] <= 0:
                        continue
                    debug["full_cloud_macro_restore_time"] = float(
                        debug.get("full_cloud_macro_restore_time", 0.0)
                        + (time.time() - restore_start)
                    )
                    candidate_encode_wall_start = time.time()
                    stats = loss._encode_actual_batch(args, candidate_xyz)
                    debug["candidate_actual_wall_time"] = float(
                        debug.get("candidate_actual_wall_time", 0.0)
                        + (time.time() - candidate_encode_wall_start)
                    )
                    debug["candidate_actual_encode_time"] = float(
                        debug.get("candidate_actual_encode_time", 0.0)
                        + float(stats.get("encode_time", 0.0) or 0.0)
                    )
                    tested += 1
                    full_cloud_macro_tested += 1
                    cand_bit = float(stats.get("bit", 0.0))
                    full_drop_count = int(
                        full_macro_item.get(
                            "drop_count",
                            len(drop_key_set) if drop_key_set is not None else 0,
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
                    keep_local = torch.ones((unique_coords.shape[0],), device=unique_coords.device, dtype=torch.bool)
                    if local_unique_indices:
                        keep_local[
                            torch.as_tensor(local_unique_indices, device=unique_coords.device, dtype=torch.long)
                        ] = False
                    local_candidate_coords = torch.unique(unique_coords[keep_local], dim=0, sorted=True)
                    local_proxy_start = time.time()
                    proxy_bits, proxy_percent = _sparsepcgc_proxy_delta_percent(
                        local_candidate_coords,
                        args,
                        base_proxy_bits,
                    )
                    debug["full_cloud_macro_local_proxy_time"] = float(
                        debug.get("full_cloud_macro_local_proxy_time", 0.0)
                        + (time.time() - local_proxy_start)
                    )
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
                    _update_best(raw_percent, actual_percent, cand_percent, edit_record_bits, cand_bit, proxy_percent)
                    if cand_percent < -min_improve:
                        full_cloud_macro_improving_count += 1
                        improving_candidate_count += 1
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
                                "override_final_voxel_coords": candidate_coords.detach().clone(),
                                "override_scope": "full_cloud",
                                "actual_stats": dict(stats),
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
    """Normalize only operation heads backed by a fresh actual-oracle label."""
    debug = {}
    if not bool(getattr(args, "repair_balance_operation_head_grads", True)):
        return debug
    target = max(float(getattr(args, "repair_operation_head_grad_target", 1.0)), 0.0)
    if target <= 0.0:
        return debug
    structure_debug = structure_debug if isinstance(structure_debug, dict) else {}

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
    min_scale = max(float(getattr(args, "repair_operation_head_grad_min_scale", 1e-4)), 0.0)
    max_scale = max(
        float(getattr(args, "repair_operation_head_grad_max_scale", 100.0)),
        min_scale,
    )
    for label, (operation, modules) in groups.items():
        if not teacher_active.get(operation, False):
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
            grad = torch.nan_to_num(param.grad.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            norm_sq += float(torch.sum(grad * grad).cpu())
        norm_before = math.sqrt(max(norm_sq, 0.0))
        if not math.isfinite(norm_before) or norm_before <= 1e-12:
            debug[f"{label}_grad_balance_status"] = "zero_or_nonfinite"
            continue
        scale = min(max(target / norm_before, min_scale), max_scale)
        for param in params:
            param.grad.mul_(float(scale))
        debug[f"{label}_grad_norm_before_balance"] = float(norm_before)
        debug[f"{label}_grad_balance_scale"] = float(scale)
        debug[f"{label}_grad_norm_after_balance"] = float(norm_before * scale)
        debug[f"{label}_grad_balance_status"] = "scaled"
    return debug


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
            "_current_subtree_id",
            "_log_this_step",
            "_collect_structure_debug",
            "_collect_sparsepcgc_debug",
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
                pts = dataset[idx]
                cache_key = f"{make_step_cache_key(file_path, args)}|episode_full_cloud_validation"
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
                try:
                    input_pcd = prepare_subtree_input_pcd(pts, use_cuda)
                    input_xyz = input_pcd[:, :3, :]
                    input_attr = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with torch.no_grad(), autocast_ctx:
                        full_octree_context = _build_full_cloud_octree_context_for_train(
                            input_xyz,
                            args,
                            coord_scale=None,
                        )
                        full_cloud_canonical_context = full_octree_context
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
    total_train_files = sum(len(dataset) for _, dataset in seq_datasets) # 全シーケンスに含まれる点群ファイル数を合計し、総Step数の見積もりなどに使用
    args._total_train_steps_estimate = max(int(getattr(args, "episodes", 1)), 1) * max(int(total_train_files), 1) # Episode数と点群ファイル数からそう学修Step数を概算
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
    for_better_path = init_for_better_logger(args, plot, writer) # 改善・改悪要因を記録する詳細分析ログ
    checkpoint_gate_refs = {} # ChackPoint保存判定で使う基準値や過去値を保持
    best_trackers = None # 複数指標でBest CheckPointを追跡するための状態を初期化
    actual_guard_state = {"best_delta": float("inf"), "best_path": None, "bad_count": 0} # 実Codex評価が悪化したときに、巻き戻す
    full_cloud_correction_state = {}
    last_subtree_actual_debug_for_correction = {}
    last_full_context_debug_for_correction = {}
    last_full_cloud_correction_update_debug = {}

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
    optimizer.zero_grad(set_to_none=True) # 本学習開始前にOptimizer内の勾配を削除

    """==========================================================="""
    """トレーニング"""
    """==========================================================="""
    prev_stage = None
    global_train_step = 0
    global_epoch = 0
    scheduler_step_count = 0
    for episode in range(args.episodes): # Episode開始
        writer.write(f"◆◆◆ Episode {episode + 1} / {args.episodes} ◆◆◆")

        """Stage変更"""
        current_stage = resolve_compression_fixed_stage(args) # EpisodeでStageを切り替えず、圧縮損失が常に効くjoint Stageへ固定する
        args.training_stage = current_stage
        if current_stage != prev_stage: # 前EpisodeとStageが異なる場合
            stage_factors = stage_loss_factors(args) # 現在Stageでっ各損失をどの比率で扱うか取得する
            writer.write(f"Training Stage Switch: episode={episode + 1}, stage={current_stage}")
            writer.write( "Stage Loss Factors: " f"geom={stage_factors['geom']}, com={stage_factors['com']}, " f"attr={stage_factors['attr']}, policy={stage_factors['policy']}, repair={stage_factors['repair']}")
            log_for_better_event( for_better_path, "stage_switch", episode=episode + 1, stage=current_stage, stage_factors=stage_factors)
            prev_stage = current_stage

        model.train()

        """変数の初期化"""
        episode_metric_sums = None
        episode_checkpoint_sums = new_checkpoint_metric_sum()
        episode_compression_sums = new_compression_episode_sum()
        episode_operation_sums = new_operation_episode_sum()
        episode_optimizer_total_count = 0
        episode_optimizer_step_count = 0
        episode_nonfinite_grad_skip_count = 0
        episode_max_consecutive_nonfinite_grad_skips = 0

        for epoch, (seq_dir, dataset) in enumerate(seq_datasets): # Epoch開始
            writer.write(f"⦿⦿⦿ Epoch {epoch + 1}/{num_seq} : {seq_dir} ⦿⦿⦿")

            """基本情報のセットアップ"""
            active_dataset = apply_epoch_file_window(dataset, args, global_epoch) # Epochごとにmax_files件の窓を順番に進め、同じ先頭30件の反復を避ける
            loader = torch.utils.data.DataLoader(active_dataset, **loader_kwargs) # 現在Epochの窓Datasetから点群ファイルを順に読み出す
            num_steps = len(active_dataset)
            epoch_has_optimizer_step = False
            epoch_metric_sums = None

            for step, pts in enumerate(loader): # Step開始
                """基本情報のセットアップ"""
                st_step = time.time()
                optimizer.zero_grad(set_to_none=True) # 前Stepの勾配を必ず消し、条件分岐による勾配蓄積を防ぐ
                file_path = active_dataset.files[step]
                cache_key = make_step_cache_key(file_path, args) # ファイルパスと設定から一意なキーを作り、前処理結果、Codec結果、Patch情報などのキャッシュ参照に使う
                raw_pts_num = int(pts.shape[1] if pts.dim() == 3 else pts.shape[0]) # 受け取ったデータの元点数を数え、点数比較やログに使用
                subtree_mode = bool(getattr(args, "train_patch_subset_enable", False)) # Octree Subtreeの部分学修を行うか否かの判定

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

                """ログ用の変数セット"""
                args._global_train_step = int(global_train_step) # 現在の累積Step番号を保存
                args._current_sample_name = os.path.basename(str(file_path)) # teacher/debugログに点群ファイル名を残す
                args._current_teacher_scope = "full_cloud" # このStepのteacherが全点群か局所subtreeかをLoss側へ伝える初期値
                args._sparsepcgc_full_cloud_actual_primary_active = False
                args._log_this_step = False
                sparsepcgc_csv_debug = ( str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc" and bool(getattr(args, "save_compression_metric_csv", True))) # Sparse PCGC専用ログ
                operation_csv_debug = bool( getattr(args, "save_operation_metric_csv", getattr(args, "save_operation_metrics_csv", True))) # 点操作メトリクスCSVを保存するか判定し、点移動量や追加/削除などのDebug収集条件に使用
                args._collect_sparsepcgc_debug = bool(sparsepcgc_csv_debug and should_collect_sparsepcgc_hard_debug(args, log_this_step=log_this_step, profile_this_step=profile_this_step, global_step=global_train_step)) # SparsePCGCの重いhard統計は毎Stepではなく診断間隔だけ収集する
                args._collect_structure_debug = bool( log_this_step or profile_this_step or operation_csv_debug or sparsepcgc_add_experiment_active(args))
                detail_log_this_step = False
                step_timing_breakdown = {}
                step_actual_oracle_metric_debug = {}

                """学習設定"""
                if timing_enabled and use_cuda and torch.cuda.is_available(): # GPU計測のためのリセット
                    torch.cuda.reset_peak_memory_stats()

                if timing_enabled: # 時間計測が有効なら入力整形処理の開始時刻を記録
                    sync_for_timing(use_cuda) # GPUを使用している場合は、正確な時間計測のためにGPUの処理が完了するのを待つ
                    timing_data_start = time.time() # 時間計測開始
                if subtree_mode: # Subtree部分学習モード
                    input_pcd = prepare_subtree_input_pcd(pts, use_cuda) # 入力点群を間引かず全点のままSubtree分割用形式へ変換する
                    input_xyz = input_pcd[:, :3, :] # 座標情報のみ抽出
                elif args.split2patch: # パッチ分割モード
                    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0) # 形式変換
                    input_pcd = downsample_input_batch(input_pcd, args, cache_key) # 点数の制限
                    if use_cuda:
                        input_pcd = input_pcd.cuda(non_blocking=True)
                    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous() # 形式変換
                    input_xyz = input_pcd[:, :3, :] # 座標情報のみ抽出
                else:
                    input_xyz, patches, centroid_xyz, fd_xyz = prepare_whole_cloud_inputs(pts, args, cache_key, use_cuda) # 全体点群を入力とする
                    input_pcd = input_xyz

                pcd_pts_num = input_xyz.shape[-1]
                # ============================================================
                # このStepで使う唯一の voxel 座標系を full cloud から一度だけ作る。
                # Subtree / full anchor / actual / proxy / debug は必ずこれを基準にする。
                # ============================================================
                full_cloud_canonical_start = time.time()
                full_cloud_canonical_context = _build_full_cloud_octree_context_for_train(
                    input_xyz[:, :3, :],
                    args,
                    coord_scale=None,
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
                if compression_primary_mode and not bool(getattr(args, "cp_use_stage_factors", False)):
                    stage_factors = {name: 1.0 for name in stage_factors} # 全Stage係数を全て1.0にする
                compute_compression = True # StageやModeに関係なく毎Stepで圧縮損失を計算する
                actual_refresh_interval = max(int(getattr(args, "actual_eval_interval", 0)), 0)
                refresh_actual_gen = bool(
                    global_train_step == 0
                    or (actual_refresh_interval > 0 and global_train_step % actual_refresh_interval == 0)
                ) # 実Codec/Surrogateの出力側更新は間引いて計算時間を抑える

                """変数の初期化と設定"""
                subset_step = False # 部分学習か否か
                subset_enabled = False # 部分集合学習が有効か否か
                is_anchor_step = True # 初期状態では全体点ん群を使うAnchor Stepとする
                compression_cache_key = cache_key # キャッシュキーの初期化
                compression_gt_pts = input_xyz # 圧縮損失で比較する教師側点群を入力点群にする
                compression_gen_xyz = None # 圧縮Lossへ渡した出力点群をVoxel衝突ログで参照する
                train_edit_stats = None # 点操作を見計算状態にする
                noise_debug = empty_noise_debug() # 圧縮損失用に量子化前の点群に加えるノイズのデバッグ情報を初期化
                # VoxelCollisionログ用のGT点群である。
                # full-cloud時は全体点群、Subtree時は後で選択Subtreeに差し替える。
                voxel_collision_input_gt = input_xyz[:, :3, :]
                subtree_depth_meta = {} # 深度などの情報保存
                subtree_trees = {} # 事前構築したSubtree内部OctreeをCPU側で保持する
                full_octree_contexts = {} # full-cloud Octree内でのSubtree文脈をCPU側で保持する
                group_meta = {} # 追加Octreeメタ情報を既存group_stateと分離して保持する
                total_subtree_count = 0 # Subtree総数を0で初期化
                eligible_subtree_count = 0 # 学習対象候補として残ったSubtreeの初期化
                actual_eligible_subtree_count = 0 # 条件を満たしたSubtreeを初期化
                selected_subtree_count = 0 # このStepで実際にForwardとLoss計算に用いるSubtreeを初期化
                min_subtree_points = 0 # Subtreeとしてさいようするための最小点数条件を初期化
                subtree_point_counts = [int(input_xyz.shape[-1])] # Subtree点数分布の初期値として、全体点群の点数をリストで保存
                anchor_reason = "not_subtree_mode"
                subtree_loss_scope = "full_cloud"
                """Subtree分割学習"""
                if subtree_mode:                    
                    """Subtree分割学習のセットアップ"""
                    optimizer.zero_grad(set_to_none=True) # 残った勾配の削除
                    subset_enabled = True # 部分集合学習を有効にする
                    input_attr_full = input_pcd[:, 3:, :].contiguous() if input_pcd.shape[1] > 3 else None # 属性のとりだし
                    subtree_depth_meta = sample_train_subtree_depth(
                        input_xyz,
                        args,
                        global_step=global_train_step,
                        cache_key=cache_key,
                    ) # Octree深度の決定

                    subtree_depth_meta, requested_subtree_depth = maybe_raise_subtree_depth_for_large_input(
                        subtree_depth_meta,
                        raw_pts_num,
                        args,
                    ) # 大点群時のSubtree深度調整

                    # Subtree分割の最小Depthを2に固定する
                    train_subtree_depth_floor = int(getattr(args, "train_subtree_depth_floor", 4))
                    requested_subtree_depth = max(int(requested_subtree_depth), train_subtree_depth_floor)

                    # ログ上も、最終的に要求したDepthを分かるように残す
                    subtree_depth_meta = dict(subtree_depth_meta)
                    subtree_depth_meta["depth_floor"] = int(train_subtree_depth_floor)
                    subtree_depth_meta["depth_after_floor"] = int(requested_subtree_depth)

                    min_subtree_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1)
                    subtree_group_build_start = time.time()
                    subtree_group_state = build_octree_subtree_groups_with_retry(
                        input_xyz,
                        args,
                        requested_subtree_depth,
                        min_subtree_points,
                        allow_largest_fallback=True,
                    )
                    step_timing_breakdown["subtree_group_build_time"] = float(time.time() - subtree_group_build_start)
                    # subtree_depth_meta = sample_train_subtree_depth( input_xyz, args, global_step=global_train_step, cache_key=cache_key) # Octree深度の決定
                    # subtree_depth_meta, requested_subtree_depth = maybe_raise_subtree_depth_for_large_input( subtree_depth_meta, raw_pts_num, args) # 大点群時は点を捨てずにSubtree深度だけ1段階浅くする
                    # requested_subtree_depth = int(requested_subtree_depth) # 調整後のSubtree深度を整数で取り出す
                    # min_subtree_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1) # Subtreeとして採用する点数の最小点数
                    # subtree_group_state = build_octree_subtree_groups_with_retry( input_xyz, args, requested_subtree_depth, min_subtree_points, allow_largest_fallback=True) # 入力点群から指定深度のOctree Subtree群を作る
                    """Subtree情報"""
                    subtree_ref = subtree_group_state["subtree_ref"] # Subtree参照情報の抽出
                    if subtree_ref is None:
                        raise RuntimeError("Subtree mode did not find any valid octree subtree.")
                    # subtree_trees = dict(subtree_group_state.get("subtree_trees", {}) or {}) # 追加フィールドだけを参照し、既存形式は変えない
                    # full_octree_contexts = dict(subtree_group_state.get("full_octree_contexts", {}) or {}) # full-cloud上の親・兄弟・祖先文脈
                    # group_meta = dict(subtree_group_state.get("group_meta", {}) or {}) # ログ用の軽量メタ情報
                    subtree_trees = {}
                    full_octree_contexts = {}
                    group_meta = {}
                    subtree_depth_meta = dict(subtree_depth_meta) # 深度メタ情報変換
                    subtree_depth_meta["requested_depth"] = int(requested_subtree_depth) # 要求された深度情報の保存
                    subtree_depth_meta["depth"] = int(subtree_group_state.get("depth", requested_subtree_depth)) # 実際に採用された深度情報の保存
                    subtree_depth_meta["retry_count"] = int(subtree_group_state.get("retry_count", 0)) # 深度変更の再試行が何回行われたか
                    subtree_depth_meta["selection_reason"] = str(subtree_group_state.get("selection_reason", "none")) #  なぜその深度・Subtreeが選ばれたか
                    all_subtree_keys = subtree_group_state["unique_keys"] # 入力点群内に存在する全Subtreeの識別Keyを取り出す
                    subtree_index_lists = subtree_group_state["index_lists"] # 各Subtree内の点インデックス
                    all_groups = subtree_group_state["all_groups"] # 全Subtreeに関して、情報を抜き出す

                    """Subtree決定"""
                    total_subtree_count = int(all_subtree_keys.numel()) # 入力点群から作られたSubtreeの総数を数える
                    eligible_groups = list(subtree_group_state.get("eligible_groups", [])) # 最小点数条件などを満たした学習候補Subtreeを取り出す
                    actual_eligible_subtree_count = int(len(eligible_groups)) # 条件を満たしたSubtreeを数える
                    min_points_miss = bool(total_subtree_count > 0 and not eligible_groups and min_subtree_points > 1) # Subtree自体はあるが、最小点数条件を満たすSubtreeがないかを判定する
                    candidate_groups = eligible_groups or list(subtree_group_state.get("groups", [])) or all_groups # 学習に使う候補Subtree集合を決める
                    max_subtree_points = max(int(getattr(args, "train_subtree_max_points", 0)), 0)
                    if max_subtree_points > 0:
                        bounded_candidate_groups = [
                            (subtree_key, point_idx)
                            for subtree_key, point_idx in candidate_groups
                            if int(point_idx.numel()) <= max_subtree_points
                        ]
                        if bounded_candidate_groups:
                            candidate_groups = bounded_candidate_groups
                    candidate_subtree_keys = all_subtree_keys.new_tensor( # 候補SubtreeのKeyを元のSubtree Keyと同じテンソルとして作る
                        [subtree_key for subtree_key, _ in candidate_groups],
                        dtype=all_subtree_keys.dtype,
                    ) if candidate_groups else all_subtree_keys.new_empty((0,), dtype=all_subtree_keys.dtype)
                    eligible_subtree_count = int(candidate_subtree_keys.numel()) # FallBack後も含めた学習候補Subtreeを数える

                    """Subtree分割学習の再セットアップ"""
                    is_anchor_step, anchor_reason = should_use_full_cloud_anchor( args, global_step=global_train_step, cache_key=cache_key) # このStepをSubtree学習が全点群学習にするか判定
                    if ( min_points_miss and eligible_subtree_count <= 0 and bool(getattr(args, "train_subtree_anchor_on_min_points_miss", False))): # 最小点群数を満たすSubtreeがない
                        is_anchor_step = True
                        anchor_reason = "min_points_miss_full_anchor"
                        log_for_better_event( for_better_path, "subtree_min_points_miss", global_step=global_train_step + 1, sampled_depth=int(subtree_depth_meta["depth"]), min_subtree_points=min_subtree_points, total_subtree_count=total_subtree_count, action="full_anchor")
                    elif min_points_miss:
                        log_for_better_event( for_better_path, "subtree_min_points_miss", global_step=global_train_step + 1, sampled_depth=int(subtree_depth_meta["depth"]), min_subtree_points=min_subtree_points, total_subtree_count=total_subtree_count, action="legacy_all_subtrees_fallback")
                    full_cloud_anchor_shadow_train_requested = bool(
                        is_anchor_step
                        and bool(getattr(args, "full_cloud_anchor_train_shadow_subtree", True))
                        and eligible_subtree_count > 0
                    )
                    selected_subtree_for_grad = bool((not is_anchor_step) or full_cloud_anchor_shadow_train_requested)
                    selected_subtree_keys = candidate_subtree_keys # 初期状態では候補Subtreeを全て選択対象にする
                    subtree_potential_select_meta = {"enabled": False, "reason": "not_evaluated"}
                    if eligible_subtree_count > 0 and selected_subtree_for_grad:
                        subtree_potential_start = time.time()
                        potential_selected_keys, subtree_potential_select_meta = _select_sparsepcgc_potential_subtree_key(
                            candidate_groups,
                            candidate_subtree_keys,
                            full_cloud_canonical_context,
                            args,
                            global_train_step,
                            cache_key,
                        )
                        step_timing_breakdown["subtree_potential_select_time"] = float(time.time() - subtree_potential_start)
                        if torch.is_tensor(potential_selected_keys) and int(potential_selected_keys.numel()) > 0:
                            selected_subtree_keys = potential_selected_keys
                        else:
                            selected_subtree_keys = select_octree_subtree_keys(candidate_subtree_keys, global_train_step, args)
                            selected_subtree_keys = select_single_subtree_key( candidate_subtree_keys, selected_subtree_keys, global_train_step, args, cache_key) # 1StepでForwardするSubtreeをランダムに1個へ絞る
                    selected_subtree_count = int(selected_subtree_keys.numel()) # 実際に選択されたSubtree数を数える
                    subset_step = selected_subtree_for_grad and selected_subtree_count < eligible_subtree_count # 候補の一部だけを使ったStepか否かの判定
                    encoder_debug_chunks = [] if detail_log_this_step else None # 詳細ログ対象Stepなら、各Subtree Forward時のEncoder Debugを保存するリスト

                    """Selected Groupsの作成"""
                    selected_groups = None
                    if selected_subtree_for_grad: # 通常SubtreeまたはFullCloud anchorのshadow学習Subtree
                        selected_key_set = set(selected_subtree_keys.detach().cpu().tolist()) # 選択されたSubtree Keyの集合
                        group_source = candidate_groups # 選択元となるSubtreeグループ集合
                        selected_groups = [ (subtree_key, point_idx) for subtree_key, point_idx in group_source if subtree_key in selected_key_set] # 選択されたSubtree Keyに対応する情報の抽出
                        if not selected_groups and group_source:
                            selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
                        if not selected_groups:
                            raise RuntimeError("Subtree mode did not select any subtree group.")
                    if is_anchor_step: # Anchorのとき
                        subtree_point_counts = [int(point_idx.numel()) for _, point_idx in (eligible_groups or [])] # 候補Subtreeの点数分布を記録するための一覧
                        if not subtree_point_counts:
                            subtree_point_counts = [int(input_xyz.shape[-1])]
                        subtree_loss_scope = "full_cloud_output_vs_full_cloud_input"
                    else:
                        subtree_point_counts = [int(point_idx.numel()) for _, point_idx in selected_groups]
                        subtree_loss_scope = "subtree_output_vs_subtree_input"
                        
                    # ============================================================
                    # 選択済みSubtreeだけmetadataを構築する
                    # これにより、同一階層の全Subtreeに対するtree/context構築を避ける
                    # ============================================================
                    if selected_subtree_for_grad:
                        t1 = time.time()
                        subtree_trees, full_octree_contexts, group_meta = build_selected_group_octree_metadata(
                            input_xyz,
                            subtree_ref,
                            selected_groups,
                            args=args,
                        )
                        # ============================================================
                        # build_selected_group_octree_metadata() が内部で局所再量子化していても、
                        # ここで必ず full cloud canonical voxel coords に差し替える。
                        # ============================================================
                        patched_subtree_trees = {}
                        patched_full_octree_contexts = {}

                        for selected_key, selected_point_idx in selected_groups:
                            selected_key_int = int(selected_key)

                            patched_subtree_tree, patched_full_context = _inject_full_cloud_canonical_into_subtree_metadata(
                                subtree_tree=subtree_trees.get(selected_key_int, {}),
                                full_octree_context=full_octree_contexts.get(selected_key_int, {}),
                                full_cloud_canonical_context=full_cloud_canonical_context,
                                point_idx=selected_point_idx,
                                device=input_xyz.device,
                            )

                            oracle_depth = 0
                            try:
                                oracle_depth = int(subtree_ref["depth"][0].item())
                            except Exception:
                                oracle_depth = int(subtree_depth_meta.get("depth", 0))
                            oracle_cache_key = (
                                f"{cache_key}|subtree_depth={oracle_depth}|subtree_key={selected_key_int}"
                            )
                            patched_full_context["actual_oracle_full_cloud_cache_key"] = str(cache_key)
                            oracle_subtree_xyz = input_xyz.index_select(2, selected_point_idx).contiguous()
                            patched_subtree_tree, patched_full_context, oracle_debug = _attach_sparsepcgc_actual_oracle_drop(
                                args=args,
                                writer=writer,
                                loss=loss,
                                subtree_tree=patched_subtree_tree,
                                full_octree_context=patched_full_context,
                                subtree_xyz=oracle_subtree_xyz[:, :3, :],
                                cache_key=oracle_cache_key,
                                global_step=global_train_step,
                            )

                            # A full-cloud structured candidate is the final output teacher, while
                            # the selected subtree only receives the intersecting local drop mask.
                            # Applying full-cloud coordinates inside the shadow subtree would mix
                            # coordinate scopes and duplicate the entire cloud in that forward.
                            if str(oracle_debug.get("override_scope", "")) == "full_cloud":
                                apply_full_override = bool(
                                    getattr(args, "sparsepcgc_actual_oracle_apply_full_override", False)
                                )
                                if apply_full_override:
                                    full_cloud_canonical_context = dict(full_cloud_canonical_context)
                                    for oracle_key, oracle_value in patched_full_context.items():
                                        if oracle_key.startswith("actual_oracle_"):
                                            full_cloud_canonical_context[oracle_key] = oracle_value
                                patched_subtree_tree = dict(patched_subtree_tree)
                                patched_full_context = dict(patched_full_context)
                                for oracle_key in (
                                    "actual_oracle_override_final_voxel_coords",
                                    "actual_oracle_override_move_count",
                                    "actual_oracle_override_drop_count",
                                    "actual_oracle_override_subtree_prune_count",
                                    "actual_oracle_override_scope",
                                ):
                                    patched_subtree_tree.pop(oracle_key, None)
                                    patched_full_context.pop(oracle_key, None)
                                patched_full_context.pop("actual_oracle_cached_edited_actual_stats", None)

                            patched_subtree_trees[selected_key_int] = patched_subtree_tree
                            patched_full_octree_contexts[selected_key_int] = patched_full_context
                            if isinstance(oracle_debug, dict) and oracle_debug:
                                step_actual_oracle_metric_debug = dict(oracle_debug)

                            if selected_key_int in group_meta:
                                group_meta[selected_key_int] = dict(group_meta[selected_key_int])
                            else:
                                group_meta[selected_key_int] = {}

                            group_meta[selected_key_int]["canonical_source"] = "full_cloud_canonical"
                            group_meta[selected_key_int]["canonical_subtree_points"] = int(
                                patched_subtree_tree["global_voxel_coords"].shape[-1]
                            )
                            group_meta[selected_key_int]["canonical_full_points"] = int(
                                patched_full_context["full_global_voxel_coords"].shape[-1]
                            )
                            if isinstance(oracle_debug, dict):
                                group_meta[selected_key_int]["actual_oracle_enabled"] = bool(
                                    oracle_debug.get("enabled", False)
                                )
                                group_meta[selected_key_int]["actual_oracle_used"] = bool(
                                    oracle_debug.get("used", False)
                                )
                                group_meta[selected_key_int]["actual_oracle_best_percent"] = float(
                                    oracle_debug.get("best_percent", 0.0)
                                )
                                group_meta[selected_key_int]["actual_oracle_reason"] = str(
                                    oracle_debug.get("reason", "")
                                )

                        subtree_trees = patched_subtree_trees
                        full_octree_contexts = patched_full_octree_contexts
                        t2 = time.time()
                        step_timing_breakdown["selected_metadata_oracle_time"] = float(t2 - t1)
                        # print(t2-t1)
                    else:
                        subtree_trees = {}
                        full_octree_contexts = {}
                        group_meta = {}

                    """ログ"""
                    if (
                        log_this_step
                        and bool(getattr(args, "train_patch_subset_log", True))
                        and not compact_step_text_log
                    ):
                        if is_anchor_step:
                            point_counts = list(subtree_point_counts)
                            stat_groups = eligible_groups or [(0, torch.arange(input_xyz.shape[-1], device=input_xyz.device))]
                            loss_scope = subtree_loss_scope
                        else:
                            point_counts = list(subtree_point_counts)
                            stat_groups = selected_groups
                            loss_scope = subtree_loss_scope
                        mean_points = sum(point_counts) / float(max(len(point_counts), 1))
                        octree_stat = summarize_subtree_octree_stats(input_xyz, stat_groups, args)
                        octree_stat_text = ""
                        if octree_stat is not None:
                            octree_stat_text = (
                                f", octree_node[min/mean/max]={octree_stat['node']}, "
                                f"octree_single[min/mean/max]={octree_stat['single']}, "
                                f"octree_depth[min/mean/max]={octree_stat['depth']}, "
                                f"octree_stat_count={int(octree_stat['count'])}"
                            )
                        potential_text = ""
                        if isinstance(subtree_potential_select_meta, dict) and bool(
                            subtree_potential_select_meta.get("enabled", False)
                        ):
                            potential_text = (
                                f", potential_priority=True"
                                f"(reason={subtree_potential_select_meta.get('reason', 'n/a')}, "
                                f"pool={int(subtree_potential_select_meta.get('pool', 0))}, "
                                f"scored={int(subtree_potential_select_meta.get('scored', 0))}, "
                                f"rank={int(subtree_potential_select_meta.get('rank', -1))}, "
                                f"score={float(subtree_potential_select_meta.get('score', 0.0)):.4f}, "
                                f"best={float(subtree_potential_select_meta.get('best_score', 0.0)):.4f}, "
                                f"drop={float(subtree_potential_select_meta.get('drop_score', 0.0)):.4f}, "
                                f"add={float(subtree_potential_select_meta.get('add_score', 0.0)):.4f}, "
                                f"macro={float(subtree_potential_select_meta.get('macro_density_score', 0.0)):.4f}, "
                                f"proxy_rate={float(subtree_potential_select_meta.get('proxy_rate_score', 0.0)):.4f}, "
                                f"fast_diag_local={int(subtree_potential_select_meta.get('fast_diag_local_count', 0))}, "
                                f"fast_diag_global={int(subtree_potential_select_meta.get('fast_diag_global_drop_count', 0))}, "
                                f"random={bool(subtree_potential_select_meta.get('random', False))})"
                            )
                        writer.write(
                            "SubtreeSelection: "
                            f"depth={int(subtree_depth_meta['depth'])} "
                            f"(base={int(subtree_depth_meta['base_depth'])}, "
                            f"range={int(subtree_depth_meta['min_depth'])}-{int(subtree_depth_meta['max_depth'])}, "
                            f"uncapped_range={int(subtree_depth_meta.get('uncapped_min_depth', subtree_depth_meta['min_depth']))}-"
                            f"{int(subtree_depth_meta.get('uncapped_max_depth', subtree_depth_meta['max_depth']))}, "
                            f"requested={int(subtree_depth_meta.get('requested_depth', subtree_depth_meta['depth']))}, "
                            f"retry={int(subtree_depth_meta.get('retry_count', 0))}, "
                            f"retry_reason={subtree_depth_meta.get('selection_reason', 'none')}, "
                            f"curriculum_phase={float(subtree_depth_meta.get('curriculum_phase', 1.0)):.3f}, "
                            f"data_max={int(subtree_depth_meta['data_max_depth'])}, "
                            f"percent_mode={bool(subtree_depth_meta.get('depth_percent_curriculum', False))}, "
                            f"percent_range={subtree_depth_meta.get('depth_percent_range', 'n/a')}), "
                            f"selected={selected_subtree_count}/{eligible_subtree_count} eligible "
                            f"(total={total_subtree_count}, min_points={min_subtree_points}), "
                            f"points[min/mean/max]={min(point_counts)}/{mean_points:.1f}/{max(point_counts)}, "
                            f"anchor_refresh={bool(is_anchor_step)}({anchor_reason}), "
                            f"loss_scope={loss_scope}"
                            f"{octree_stat_text}"
                            f"{potential_text}"
                        )

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
                    L_full_context_subtree_delta = input_xyz.new_zeros(())
                    full_context_subtree_delta_debug = {}
                    full_cloud_correction_loss = input_xyz.new_zeros(())
                    full_cloud_correction_debug = {}
                    gen_xyz = None
                    final_w = None
                    out_label = None
                    full_cloud_anchor_no_grad = False
                    full_cloud_anchor_no_grad_reason = ""
                    full_cloud_anchor_shadow_train_active = False
                    full_cloud_anchor_debug_snapshot = {}
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
                        ) # このSubtree処理内で詳細ログを出すか否か決定
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
                            full_cloud_anchor_shadow_train_active = bool(
                                full_cloud_anchor_shadow_train_requested
                                and full_cloud_anchor_no_grad
                                and selected_groups
                            )
                            if full_cloud_anchor_shadow_train_requested and not compact_step_text_log:
                                writer.write(
                                    "FullCloudAnchorShadowTrain: "
                                    f"requested={bool(full_cloud_anchor_shadow_train_requested)}, "
                                    f"active={bool(full_cloud_anchor_shadow_train_active)}, "
                                    f"selected_subtrees={int(len(selected_groups or []))}, "
                                    f"reason={'active_shadow_subtree_grad' if full_cloud_anchor_shadow_train_active else 'full_cloud_grad_allowed_or_no_subtree'}"
                                )

                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                            grad_ctx = torch.no_grad() if full_cloud_anchor_no_grad else nullcontext()

                            with grad_ctx, autocast_ctx: # 全体点群をno-gradでモデルに入力し、teacher更新用の出力だけ得る
                                """モデルの実行"""
                                # Step冒頭で作った full cloud canonical context をそのまま使う。
                                # ここで再量子化してはいけない。
                                full_octree_context = dict(full_cloud_canonical_context)
                                full_octree_context["octree_context_scope"] = "full_cloud"
                                full_octree_context["octree_input_mode"] = "full_cloud"
                                full_octree_context["canonical_source"] = "full_cloud_canonical"
                                full_octree_context["fast_full_cloud_oracle_anchor"] = bool(
                                    full_cloud_anchor_shadow_train_active
                                    and bool(getattr(args, "sparsepcgc_actual_oracle_apply_full_override", False))
                                    and isinstance(step_actual_oracle_metric_debug, dict)
                                    and bool(step_actual_oracle_metric_debug.get("used", False))
                                    and str(step_actual_oracle_metric_debug.get("override_scope", "")) == "full_cloud"
                                )
                                gen_pts, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward(
                                    input_xyz,
                                    input_attr_full,
                                    cache_key=cache_key,
                                    return_attr_output=False,
                                    compute_internal_losses=not bool(full_cloud_anchor_no_grad),
                                    subtree_ref=subtree_ref,
                                    selected_subtree_keys=None,
                                    subtree_tree=None,
                                    full_octree_context=full_octree_context,
                                    octree_input_mode="full_cloud",
                                )
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
                            if full_cloud_anchor_shadow_train_active:
                                train_edit_stats = None
                            else:
                                train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 入力点群と出力点群を比較し、各操作の編集統計を計算
                            final_w_for_loss = None
                            if _discrete_loss_mode_value(args) != "hard":
                                final_w_for_loss = final_w
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # 形状損失と圧縮損失の計算もAMP文脈で行うための設定を作る
                            loss_grad_ctx = torch.no_grad() if full_cloud_anchor_no_grad else nullcontext()

                            with loss_grad_ctx, autocast_ctx:
                                """形状損失の計算"""
                                full_cloud_oracle_fast_path = bool(
                                    full_cloud_anchor_shadow_train_active
                                    and isinstance(out_label, dict)
                                    and bool(out_label.get("full_cloud_oracle_fast_path", False))
                                )
                                if full_cloud_oracle_fast_path:
                                    L_geom = input_xyz.new_zeros(())
                                else:
                                    L_geom = loss.get_geometry_loss(
                                        args,
                                        gen_pts=gen_xyz,
                                        gt_pts=input_xyz[:, :3, :],
                                        final_w=final_w_for_loss,
                                        out_label=out_label,
                                    )

                                """圧縮損失の計算"""
                                if stage_factors["com"] != 0.0:
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
                                        subtree_tree=None,
                                        full_octree_context=full_octree_context,
                                        octree_input_mode="full_cloud",
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
                                            billed_tensor = L_com.new_tensor(float(billed_percent))
                                            L_com = billed_tensor + (L_com - L_com.detach())
                                            loss_bit = billed_tensor + (loss_bit - loss_bit.detach())
                                            billed_debug.update(
                                                {
                                                    "total_bit": float(billed_percent),
                                                    "actual_total_bit_percent": float(billed_percent),
                                                    "actual_train_objective_percent": float(billed_percent),
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
                                                    "actual_target": float(billed_percent),
                                                    "actual_forward_value": float(billed_percent),
                                                    "compression_forward_teacher_percent": float(billed_percent),
                                                    "forward_display_value": float(billed_percent),
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
                                    base_model_for_correction = model.module if hasattr(model, "module") else model
                                    full_cloud_debug_for_correction = dict(getattr(loss, "last_compression_debug", {}) or {})
                                    full_cloud_correction_state, last_full_cloud_correction_update_debug = update_full_cloud_actual_correction_state(
                                        args=args,
                                        state=full_cloud_correction_state,
                                        full_cloud_debug=full_cloud_debug_for_correction,
                                        subtree_debug=last_subtree_actual_debug_for_correction,
                                        full_context_debug=last_full_context_debug_for_correction,
                                        actuator_voxel_state=getattr(base_model_for_correction, "last_actuator_voxel_state", None),
                                        reference=gen_xyz_for_actual,
                                        global_step=global_train_step,
                                    )
                                    try:
                                        setattr(args, "_full_cloud_actual_correction_state", full_cloud_correction_state)
                                    except Exception:
                                        pass
                                else:
                                    writer.write("!!! Skipping compression loss calculation due to stage factor setting. !!!")
                                    zero = input_xyz.new_zeros(())
                                    L_com = zero
                                    loss_bit = zero
                                    loss_single = zero
                                    loss_nodes = zero
                            if full_cloud_anchor_shadow_train_active:
                                full_cloud_anchor_debug_snapshot = dict(getattr(loss, "last_compression_debug", {}) or {})
                                zero = input_xyz.new_zeros(())
                                L_geom = zero
                                L_com = zero
                                L_attr = zero
                                L_policy = zero
                                L_actuator = zero
                                Lp_out = zero
                                La_fit = zero
                                La_rep = zero
                                loss_bit = zero
                                loss_single = zero
                                loss_nodes = zero
                                L_full_context_subtree_delta = zero
                                full_context_subtree_delta_debug = {}
                                noise_debug = {}
                                train_edit_stats = None
                                gen_xyz = None
                                final_w = None
                                out_label = None
                                loss.last_compression_terms = {}
                                if not compact_step_text_log:
                                    writer.write(
                                        "FullCloudAnchorShadowTrain: "
                                        "full-cloud actual/correction state updated; "
                                        "resetting differentiable losses and running selected subtree with grad."
                                    )
                            step_timing_breakdown["full_cloud_anchor_block_time"] = float(
                                time.time() - full_cloud_anchor_block_start
                            )
                        if (not is_anchor_step) or full_cloud_anchor_shadow_train_active:
                            """Subtreeの場合"""
                            if full_cloud_anchor_shadow_train_active and not compact_step_text_log:
                                writer.write("Running shadow subtree step for FullCloud anchor gradient.") # FullCloud anchor用の軽量grad経路
                            elif not compact_step_text_log:
                                writer.write("Running subtree step with selected Subtree.") # Subtree Stepであることをログに出す
                            num_selected = float(max(len(selected_groups), 1)) # 選択されたSubtree数をFloatで取得
                            subtree_edit_sums = new_point_edit_sums() # 複数Subtreeの点編集統計を累積するための変数を初期化
                            subtree_noise_debug_values = [] # 各Subtreeで圧縮用ノイズを加えたかなどを統合
                            subtree_compression_term_sums = {} # Subtreeごとの圧縮損失内訳を累積する辞書
                            subtree_full_context_delta_debug_values = [] # 各Subtreeのfull-context delta debugを統合するためのリスト

                            for subtree_key, point_idx in selected_groups: # 選択されたSubtreeを1つずつ取り出し、それぞれ日いて点群を切り出し、Forward、形状損失、圧縮損失を計算
                                args._current_teacher_scope = "subtree_local" # Subtree stepではteacherが局所点群基準であることをLoss側へ渡す
                                args._current_subtree_id = str(subtree_key) # bad caseやteacherログでsubtree識別子を保存する
                                subtree_key_int = int(subtree_key) # 追加メタ辞書参照用にPython intへ揃える
                                subtree_tree = subtree_trees.get(subtree_key_int) # 事前構築Subtree Octreeを取得する
                                full_octree_context = full_octree_contexts.get(subtree_key_int) # full-cloud上のSubtree文脈を取得する
                                subtree_group_meta = group_meta.get(subtree_key_int, {}) or {} # ログ用メタ
                                use_subtree_tree = isinstance(subtree_tree, dict) and torch.is_tensor(subtree_tree.get("global_voxel_coords", None))
                                use_full_octree_context = isinstance(full_octree_context, dict) and torch.is_tensor(
                                    full_octree_context.get("full_global_voxel_coords", None)
                                )

                                if not use_subtree_tree or not use_full_octree_context:
                                    raise RuntimeError(
                                        "Full-cloud canonical voxel basis is required for subtree training. "
                                        f"subtree_key={subtree_key_int}, "
                                        f"use_subtree_tree={use_subtree_tree}, "
                                        f"use_full_octree_context={use_full_octree_context}"
                                    )

                                octree_input_mode = "prebuilt_subtree_tree"
                                args._current_exact_teacher_mode = (
                                    "subtree_with_global_context"
                                    if (use_subtree_tree and use_full_octree_context)
                                    else "local_subtree"
                                )
                                # args._current_exact_teacher_mode = "global_subtree" if (use_subtree_tree and use_full_octree_context) else "local_subtree" # teacherの意味を分離する
                                args._current_exact_teacher_uses_full_context = bool(use_subtree_tree and use_full_octree_context) # full文脈の使用可否
                                args._current_exact_teacher_fallback_reason = "" if (use_subtree_tree and use_full_octree_context) else "missing_prebuilt_subtree_tree_or_full_octree_context" # fallback理由
                                subtree_xyz = input_xyz.index_select(2, point_idx).contiguous() # 全体対入力点群から現在Subtreeに属する点だけを取り出す
                                # VoxelCollisionログでは、Subtree学習中のGTも選択Subtreeに揃える。
                                # これを入れないと input_gt だけ full cloud 全体になり、診断対象がずれる。
                                voxel_collision_input_gt = subtree_xyz[:, :3, :]
                                subtree_attr = None
                                if input_attr_full is not None:
                                    subtree_attr = input_attr_full.index_select(2, point_idx).contiguous() # 属性を取り出す
                                subtree_cache_key = ( f"{cache_key}|subtree_depth={int(subtree_ref['depth'][0].item())}|subtree_key={subtree_key}")
                                if log_this_step and not compact_step_text_log:
                                    selected_path = subtree_group_meta.get("subtree_path", None)
                                    root_path = full_octree_context.get("root_to_subtree_path", None) if isinstance(full_octree_context, dict) else None
                                    parent_occ = full_octree_context.get("parent_occupancy_code", None) if isinstance(full_octree_context, dict) else None
                                    sibling_ids = full_octree_context.get("sibling_node_ids", []) if isinstance(full_octree_context, dict) else []
                                    global_offset = subtree_group_meta.get("global_offset", None)
                                    local_offset = subtree_xyz[:, :3, :].detach().amin(dim=2).reshape(-1).tolist()
                                    global_depth = subtree_group_meta.get("global_depth", None)
                                    local_depth = int(subtree_group_meta.get("local_depth", subtree_depth_meta.get("depth", 0)))
                                    writer.write(
                                        "SubtreeOctreeInput: "
                                        f"use_subtree_tree={bool(use_subtree_tree)}, "
                                        f"use_full_octree_context={bool(use_full_octree_context)}, "
                                        f"octree_input_mode={octree_input_mode}, "
                                        f"structural_voxel_mode={'global_context' if use_subtree_tree else 'local_recomputed'}, "
                                        f"point_feature_voxel_mode={'global_context' if use_subtree_tree else 'local_xyz'}, "
                                        f"local_recomputed={bool(not use_subtree_tree)}, "
                                        f"selected_subtree_key={subtree_key_int}, "
                                        f"selected_subtree_path={selected_path}, "
                                        f"root_to_subtree_path={root_path}, "
                                        f"global_offset={global_offset}, "
                                        f"local_offset={local_offset}, "
                                        f"global_depth={global_depth}, "
                                        f"local_depth={local_depth}, "
                                        f"parent_occupancy_code={parent_occ}, "
                                        f"sibling_count={len(sibling_ids) if hasattr(sibling_ids, '__len__') else 0}, "
                                        f"enable_sparsepcgc_exact_occupancy_teacher={bool(getattr(args, 'enable_sparsepcgc_exact_occupancy_teacher', False))}, "
                                        f"sparsepcgc_exact_teacher_mode={args._current_exact_teacher_mode}, "
                                        f"exact_teacher_uses_full_context={bool(args._current_exact_teacher_uses_full_context)}, "
                                        f"exact_teacher_fallback_reason={args._current_exact_teacher_fallback_reason}"
                                    )
                                    if not use_subtree_tree or not use_full_octree_context:
                                        writer.write(
                                            "SubtreeOctreeContextFallback: "
                                            f"subtree_key={subtree_key_int}, "
                                            f"octree_input_mode=local_recomputed, "
                                            f"reason={args._current_exact_teacher_fallback_reason}"
                                        )
                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # Subtree Forward用のAMP文脈を作る
                                with autocast_ctx:
                                    """モデルの実行"""
                                    gen_subtree_pts, L_attr_sub, L_policy_sub, L_actuator_sub, final_w_sub, Lp_out_sub, La_fit_sub, La_rep_sub, out_label_sub = model.forward(
                                        subtree_xyz,
                                        subtree_attr,
                                        cache_key=subtree_cache_key,
                                        return_attr_output=False,
                                        subtree_tree=subtree_tree,
                                        full_octree_context=full_octree_context,
                                        octree_input_mode=octree_input_mode,
                                        )

                                """詳細のログ"""
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                                gen_subtree_xyz = gen_subtree_pts[:, :3, :]
                                base_model_for_full_context = model.module if hasattr(model, "module") else model
                                actuator_voxel_state_sub = getattr(base_model_for_full_context, "last_actuator_voxel_state", None)
                                subtree_edit_stats = summarize_point_edits( input_xyz=subtree_xyz[:, :3, :], gen_pts=gen_subtree_pts, final_w=final_w_sub, args=args) # Subtree入力とSubtree出力を比較し、操作などを計算する
                                add_point_edit_sums(subtree_edit_sums, subtree_edit_stats) # 現在Subtreeの編集統計を、Step全体の編集統計に累積する
                                final_w_sub_loss = None
                                if _discrete_loss_mode_value(args) != "hard":
                                    final_w_sub_loss = final_w_sub

                                """損失計算"""
                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # Subtree損失計算用のAMP文脈を作る
                                with autocast_ctx:
                                    """形状損失の計算"""
                                    L_geom_sub = loss.get_geometry_loss( args, gen_pts=gen_subtree_xyz, gt_pts=subtree_xyz[:, :3, :], final_w=final_w_sub_loss, out_label=out_label_sub)
                                    if stage_factors["com"] != 0.0:
                                        """圧縮損失の計算"""

                                        # ============================================================
                                        # Subtree actual/proxy/full-context/debug の基準を、
                                        # gen_subtree_xyz ではなく Actuator の final voxel edit state に統一する。
                                        # geometry loss は上で gen_subtree_xyz を使って計算済みなので変更しない。
                                        # ============================================================
                                        subtree_voxel_state_xyz, voxel_restored_actual_debug = _select_actual_gen_xyz_from_voxel_state(
                                            args,
                                            writer,
                                            model,
                                            gen_subtree_xyz,
                                            prefix="VoxelRestoredActual[subtree]",
                                            canonical_context=full_octree_context,
                                        )

                                        subtree_voxel_state_used = bool(
                                            isinstance(voxel_restored_actual_debug, dict)
                                            and voxel_restored_actual_debug.get("used", False)
                                            and not voxel_restored_actual_debug.get("fallback", False)
                                        )

                                        subtree_compression_source_xyz = subtree_voxel_state_xyz

                                        # ============================================================
                                        # 空点群ガード:
                                        # voxel復元失敗後の fallback_xyz まで空の場合、
                                        # surrogate/proxy 圧縮損失へ N=0 点群を渡すと amax/argmax で落ちる。
                                        # この場合は圧縮損失用だけ original subtree 入力へ退避する。
                                        # geometry loss は上で gen_subtree_xyz に対して計算済みなので変更しない。
                                        # ============================================================
                                        compression_source_empty = (
                                            (not torch.is_tensor(subtree_compression_source_xyz))
                                            or subtree_compression_source_xyz.ndim != 3
                                            or subtree_compression_source_xyz.shape[1] < 3
                                            or subtree_compression_source_xyz.shape[-1] <= 0
                                        )

                                        if compression_source_empty:
                                            if writer is not None and hasattr(writer, "write") and bool(getattr(args, "_log_this_step", True)):
                                                writer.write(
                                                    "SubtreeCompressionInputGuard: "
                                                    "fallback=True, "
                                                    "reason=empty_subtree_compression_source, "
                                                    f"voxel_state_used={bool(subtree_voxel_state_used)}, "
                                                    f"gen_subtree_shape={tuple(gen_subtree_xyz.shape) if torch.is_tensor(gen_subtree_xyz) else None}, "
                                                    f"subtree_xyz_shape={tuple(subtree_xyz[:, :3, :].shape) if torch.is_tensor(subtree_xyz) else None}"
                                                )

                                            # 圧縮損失用だけ安全な非空点群に戻す。
                                            # detachしておくことで、この退避経路から不自然な勾配を返さない。
                                            subtree_compression_source_xyz = subtree_xyz[:, :3, :].detach()

                                            # actual_gen_xyz 側も空にしない。
                                            subtree_voxel_state_xyz = subtree_compression_source_xyz

                                            # このstepは voxel state を使った圧縮評価ではない扱いにする。
                                            subtree_voxel_state_used = False

                                            # final_w が全0だと、ここでも再び空扱いになる可能性がある。
                                            # そのため空点群退避時は final_w を圧縮損失へ渡さない。
                                            final_w_sub_compression = None

                                            if isinstance(voxel_restored_actual_debug, dict):
                                                voxel_restored_actual_debug["fallback"] = True
                                                voxel_restored_actual_debug["reason"] = "empty_subtree_compression_source_guard"
                                                voxel_restored_actual_debug["actual_input_source"] = "original_subtree_xyz_empty_guard"
                                                voxel_restored_actual_debug["subtree_proxy_uses_voxel_state"] = False
                                                voxel_restored_actual_debug["subtree_actual_uses_voxel_state"] = False
                                                voxel_restored_actual_debug["subtree_final_w_disabled_for_voxel_state"] = True
                                        else:
                                            # voxel state 復元点群はすでに Prune/Add/Move 反映後の occupied voxel 集合である。
                                            # ここへ final_w_sub_loss をさらに渡すと、Prune が二重反映される危険がある。
                                            # fallback時だけ従来の final_w_sub_loss を使う。
                                            final_w_sub_compression = None if subtree_voxel_state_used else final_w_sub_loss

                                        compression_subtree_xyz, noise_debug_sub = prepare_compression_points(
                                            subtree_compression_source_xyz,
                                            args,
                                            model,
                                            collect_stats=bool(log_this_step or profile_this_step),
                                        )
                                        subtree_noise_debug_values.append(noise_debug_sub)

                                        # Phase7/debug/CSV用に、Subtree actual が何を見たかを保存する。
                                        if isinstance(voxel_restored_actual_debug, dict):
                                            voxel_restored_actual_debug = dict(voxel_restored_actual_debug)
                                        else:
                                            voxel_restored_actual_debug = {}

                                        voxel_restored_actual_debug.update(
                                            {
                                                "actual_scope": "subtree",
                                                "actual_input_source": "voxel_edit_state" if subtree_voxel_state_used else "gen_subtree_xyz_fallback",
                                                "voxel_restored_actual_used": bool(subtree_voxel_state_used),
                                                "voxel_restored_actual_fallback": bool(voxel_restored_actual_debug.get("fallback", False)),
                                                "voxel_restored_actual_fallback_reason": str(voxel_restored_actual_debug.get("reason", "")),
                                                "subtree_proxy_uses_voxel_state": bool(subtree_voxel_state_used),
                                                "subtree_actual_uses_voxel_state": bool(subtree_voxel_state_used),
                                                "subtree_final_w_disabled_for_voxel_state": bool(subtree_voxel_state_used),
                                            }
                                        )

                                        try:
                                            setattr(args, "_last_voxel_restored_actual_debug", dict(voxel_restored_actual_debug))
                                        except Exception:
                                            pass
                                        args._current_exact_teacher_mode = "local_subtree"
                                        args._current_exact_teacher_uses_full_context = False
                                        args._current_exact_teacher_fallback_reason = "subtree_training_step"
                                        L_com_sub, loss_bit_sub, loss_single_sub, loss_nodes_sub, _, _ = loss.get_compression_loss(
                                            args,
                                            gen_xyz=compression_subtree_xyz,
                                            gt_xyz=subtree_xyz[:, :3, :],
                                            final_w=final_w_sub_compression,
                                            cache_key=subtree_cache_key,
                                            refresh_actual_gen=refresh_actual_gen,
                                            actual_gen_xyz=subtree_voxel_state_xyz,
                                            subtree_tree=subtree_tree,
                                            full_octree_context=full_octree_context,
                                            octree_input_mode=octree_input_mode,
                                        )

                                        # loss 側の debug にも、Subtree actual 入力の情報を混ぜる。
                                        # これを入れないと、後段の Phase7 summary で used=True が見えにくい。
                                        subtree_comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                                        subtree_comp_debug.update(voxel_restored_actual_debug)
                                        loss.last_compression_debug = subtree_comp_debug

                                        L_full_context_subtree_delta_sub, full_context_subtree_delta_debug_sub = build_full_context_subtree_delta_loss(
                                            args,
                                            full_octree_context=full_octree_context,
                                            subtree_tree=subtree_tree,
                                            actuator_voxel_state=actuator_voxel_state_sub,
                                            reference=subtree_voxel_state_xyz,
                                        )
                                        L_full_context_subtree_delta = L_full_context_subtree_delta + (
                                            L_full_context_subtree_delta_sub / num_selected
                                        )
                                        if isinstance(full_context_subtree_delta_debug_sub, dict):
                                            subtree_full_context_delta_debug_values.append(full_context_subtree_delta_debug_sub)
                                        accumulate_compression_terms( subtree_compression_term_sums, getattr(loss, "last_compression_terms", {}) or {}, 1.0 / num_selected) # 現在Subtreeで計算された圧縮損失内訳を1/Subtree数の重みをつけて、Step全体の圧縮損失内訳に累積する
                                    else:
                                        zero = subtree_xyz.new_zeros(())
                                        L_com_sub = zero
                                        loss_bit_sub = zero
                                        loss_single_sub = zero
                                        loss_nodes_sub = zero
                                        full_context_subtree_delta_debug_sub = {
                                            "full_context_subtree_delta_used": False,
                                            "full_context_subtree_delta_reason": "compression_stage_disabled",
                                            "full_context_subtree_delta_value": 0.0,
                                        }
                                        subtree_full_context_delta_debug_values.append(full_context_subtree_delta_debug_sub)
                                

                                """損失項の計算"""
                                L_geom = L_geom + (L_geom_sub / num_selected)
                                L_com = L_com + (L_com_sub / num_selected)
                                L_attr = L_attr + (L_attr_sub / num_selected)
                                L_policy = L_policy + (L_policy_sub / num_selected)
                                L_actuator = L_actuator + (L_actuator_sub / num_selected)
                                Lp_out = Lp_out + (Lp_out_sub / num_selected)
                                La_fit = La_fit + (La_fit_sub / num_selected)
                                La_rep = La_rep + (La_rep_sub / num_selected)
                                loss_bit = loss_bit + (loss_bit_sub / num_selected)
                                loss_single = loss_single + (loss_single_sub / num_selected)
                                loss_nodes = loss_nodes + (loss_nodes_sub / num_selected)
                                gen_xyz = gen_subtree_xyz
                                final_w = final_w_sub
                                out_label = out_label_sub
                            train_edit_stats = finalize_point_edit_sums(subtree_edit_sums)
                            noise_debug = merge_noise_debug_values(subtree_noise_debug_values)
                            if subtree_compression_term_sums:
                                loss.last_compression_terms = subtree_compression_term_sums
                            last_subtree_actual_debug_for_correction = dict(getattr(loss, "last_compression_debug", {}) or {})
                            if subtree_full_context_delta_debug_values:
                                merged_full_context_debug = {}
                                all_keys = set()
                                for item in subtree_full_context_delta_debug_values:
                                    if isinstance(item, dict):
                                        all_keys.update(item.keys())

                                for key in sorted(all_keys):
                                    values = [
                                        item.get(key)
                                        for item in subtree_full_context_delta_debug_values
                                        if isinstance(item, dict) and key in item
                                    ]
                                    numeric_values = []
                                    bool_values = []
                                    text_values = []
                                    for value in values:
                                        if isinstance(value, bool):
                                            bool_values.append(value)
                                        elif isinstance(value, (int, float)):
                                            numeric_values.append(float(value))
                                        elif torch.is_tensor(value) and value.numel() == 1:
                                            numeric_values.append(float(value.detach().float().cpu()))
                                        elif value is not None:
                                            text_values.append(str(value))

                                    if numeric_values:
                                        merged_full_context_debug[key] = sum(numeric_values) / float(max(len(numeric_values), 1))
                                    elif bool_values:
                                        merged_full_context_debug[key] = any(bool_values)
                                    elif text_values:
                                        merged_full_context_debug[key] = "|".join(sorted(set(text_values)))

                                full_context_subtree_delta_debug = merged_full_context_debug
                            if isinstance(full_context_subtree_delta_debug, dict):
                                last_full_context_debug_for_correction = dict(full_context_subtree_delta_debug)
                            if full_cloud_anchor_shadow_train_active and full_cloud_anchor_debug_snapshot:
                                base_model_for_shadow_correction = model.module if hasattr(model, "module") else model
                                (
                                    full_cloud_correction_state,
                                    last_full_cloud_correction_update_debug,
                                ) = update_full_cloud_actual_correction_state(
                                    args=args,
                                    state=full_cloud_correction_state,
                                    full_cloud_debug=full_cloud_anchor_debug_snapshot,
                                    subtree_debug=last_subtree_actual_debug_for_correction,
                                    full_context_debug=last_full_context_debug_for_correction,
                                    actuator_voxel_state=getattr(base_model_for_shadow_correction, "last_actuator_voxel_state", None),
                                    reference=gen_xyz if torch.is_tensor(gen_xyz) else input_xyz,
                                    global_step=global_train_step,
                                )
                                if isinstance(last_full_cloud_correction_update_debug, dict):
                                    last_full_cloud_correction_update_debug = dict(last_full_cloud_correction_update_debug)
                                    last_full_cloud_correction_update_debug["full_cloud_corr_update_source"] = "full_cloud_anchor_shadow_subtree"
                                try:
                                    setattr(args, "_full_cloud_actual_correction_state", full_cloud_correction_state)
                                except Exception:
                                    pass
                                if bool(getattr(args, "sparsepcgc_full_cloud_actual_primary", True)):
                                    full_cloud_primary_value = finite_float_or_none(
                                        full_cloud_anchor_debug_snapshot.get(
                                            "actual_total_bit_percent",
                                            full_cloud_anchor_debug_snapshot.get("actual_bit_percent", None),
                                        )
                                    )
                                    subtree_primary_value = finite_float_or_none(
                                        last_subtree_actual_debug_for_correction.get(
                                            "actual_total_bit_percent",
                                            last_subtree_actual_debug_for_correction.get("total_bit", None),
                                        )
                                    )
                                    if (
                                        full_cloud_primary_value is not None
                                        and torch.is_tensor(L_com)
                                    ):
                                        full_cloud_primary_raw_policy_value = finite_float_or_none(
                                            full_cloud_anchor_debug_snapshot.get(
                                                "policy_actual_noop_guard_raw_percent",
                                                full_cloud_anchor_debug_snapshot.get("actual_raw_percent", None),
                                            )
                                        )
                                        full_cloud_primary_oracle_source = (
                                            bool(full_cloud_anchor_debug_snapshot.get("oracle_full_cloud_override_used", False))
                                            or str(full_cloud_anchor_debug_snapshot.get("policy_action_source", "")) == "actual_oracle_full_cloud_override"
                                        )
                                        full_cloud_primary_noop_guard = bool(
                                            full_cloud_anchor_debug_snapshot.get("policy_actual_noop_guard_used", False)
                                        )
                                        full_cloud_primary_is_zero = abs(float(full_cloud_primary_value)) <= 1e-9
                                        subtree_primary_has_signal = (
                                            subtree_primary_value is not None
                                            and abs(float(subtree_primary_value)) > 1e-9
                                        )
                                        raw_policy_has_signal = (
                                            full_cloud_primary_raw_policy_value is not None
                                            and abs(float(full_cloud_primary_raw_policy_value)) > 1e-9
                                        )
                                        suppress_zero_full_cloud_primary = (
                                            full_cloud_primary_is_zero
                                            and not full_cloud_primary_oracle_source
                                            and (
                                                subtree_primary_has_signal
                                                or (full_cloud_primary_noop_guard and raw_policy_has_signal)
                                            )
                                        )
                                        if suppress_zero_full_cloud_primary:
                                            full_cloud_primary_override_debug = {
                                                "full_cloud_actual_primary_used": False,
                                                "full_cloud_actual_primary_reason": "zero_full_cloud_primary_preserved_subtree_actual",
                                                "full_cloud_actual_primary_forward_value": float(full_cloud_primary_value),
                                                "full_cloud_actual_primary_subtree_forward_before": (
                                                    float(subtree_primary_value) if subtree_primary_value is not None else None
                                                ),
                                                "full_cloud_actual_primary_raw_policy_value": (
                                                    float(full_cloud_primary_raw_policy_value)
                                                    if full_cloud_primary_raw_policy_value is not None
                                                    else None
                                                ),
                                                "full_cloud_actual_primary_noop_guard_used": bool(full_cloud_primary_noop_guard),
                                                "full_cloud_actual_primary_oracle_source": bool(full_cloud_primary_oracle_source),
                                                "full_cloud_actual_primary_suppressed_zero": True,
                                                "full_cloud_actual_primary_grad_source": "shadow_subtree_ste",
                                                "full_cloud_actual_primary_requires_grad": bool(L_com.requires_grad),
                                                "full_cloud_actual_primary_grad_fn": (
                                                    type(L_com.grad_fn).__name__ if getattr(L_com, "grad_fn", None) is not None else ""
                                                ),
                                            }
                                            try:
                                                args._sparsepcgc_full_cloud_actual_primary_active = False
                                            except Exception:
                                                pass
                                        else:
                                            full_cloud_primary_tensor = L_com.new_tensor(float(full_cloud_primary_value))
                                            L_com = full_cloud_primary_tensor + (L_com - L_com.detach())
                                            if torch.is_tensor(loss_bit):
                                                loss_bit = full_cloud_primary_tensor + (loss_bit - loss_bit.detach())
                                            full_cloud_primary_override_debug = {
                                                "full_cloud_actual_primary_used": True,
                                                "full_cloud_actual_primary_reason": "active",
                                                "full_cloud_actual_primary_forward_value": float(full_cloud_primary_value),
                                                "full_cloud_actual_primary_subtree_forward_before": (
                                                    float(subtree_primary_value) if subtree_primary_value is not None else None
                                                ),
                                                "full_cloud_actual_primary_raw_policy_value": (
                                                    float(full_cloud_primary_raw_policy_value)
                                                    if full_cloud_primary_raw_policy_value is not None
                                                    else None
                                                ),
                                                "full_cloud_actual_primary_noop_guard_used": bool(full_cloud_primary_noop_guard),
                                                "full_cloud_actual_primary_oracle_source": bool(full_cloud_primary_oracle_source),
                                                "full_cloud_actual_primary_suppressed_zero": False,
                                                "full_cloud_actual_primary_grad_source": "shadow_subtree_ste",
                                                "full_cloud_actual_primary_requires_grad": bool(L_com.requires_grad),
                                                "full_cloud_actual_primary_grad_fn": (
                                                    type(L_com.grad_fn).__name__ if getattr(L_com, "grad_fn", None) is not None else ""
                                                ),
                                            }
                                            try:
                                                args._sparsepcgc_full_cloud_actual_primary_active = True
                                            except Exception:
                                                pass
                                    else:
                                        full_cloud_primary_override_debug = {
                                            "full_cloud_actual_primary_used": False,
                                            "full_cloud_actual_primary_reason": "missing_full_cloud_value_or_lcom_tensor",
                                        }
                                        try:
                                            args._sparsepcgc_full_cloud_actual_primary_active = False
                                        except Exception:
                                            pass

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
                        final_w_for_loss = final_w
                    if timing_enabled:
                        sync_for_timing(use_cuda)
                        timing_noise_start = time.time()
                    if subtree_mode:
                        compression_gen_xyz = gen_xyz
                    else: # 入力や診断前ではなく、編集後・量子化前にだけ一様ノイズを加える
                        compression_gen_xyz, noise_debug = prepare_compression_points( gen_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step)) # 出力点群から圧縮損失用点群を作る
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
                # Phase7-2のfull-context subtree deltaをここへ追加するため、
                # termsを使う前に必ず初期化する。
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
                if torch.is_tensor(L_full_context_subtree_delta):
                    terms = dict(terms)
                    terms["full_context_subtree_delta"] = L_full_context_subtree_delta
                    if isinstance(full_context_subtree_delta_debug, dict):
                        full_context_subtree_delta_debug["full_context_subtree_delta_added_to_terms"] = True
                        full_context_subtree_delta_debug["full_context_subtree_delta_requires_grad"] = bool(
                            L_full_context_subtree_delta.requires_grad
                        )
                    
                minimal_loss_objective = bool(getattr(args, "minimal_loss_objective", True))
                L_com_objective = compose_train_compression_objective(args, terms, L_com, La_fit) # minimal_loss_objective時はL_comをそのまま通す
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

                """形状損失を合成"""
                legacy_L_downstream = (
                    stage_factors["geom"] * args.w_geom * L_geom
                    + stage_factors["com"] * float(getattr(args, "w_com", 10.0)) * L_com_objective
                ) # 形状損失と圧縮損失の合成

                """属性/方策/操作損失を合成"""
                legacy_L_total = legacy_L_downstream
                if not minimal_loss_objective:
                    legacy_L_total = (
                        legacy_L_downstream
                        + stage_factors["attr"] * args.w_attr * L_attr
                        + stage_factors["policy"] * args.w_policy * L_policy
                        + stage_factors["repair"] * args.w_actuator * L_actuator
                    )

                """損失の合成"""
                L = legacy_L_total
                L_downstream = legacy_L_downstream
                L_discrete_policy = L.new_zeros(())
                cp_debug = {} # compression primaryモード用のdebug情報を空辞書で初期化
                if compression_primary_mode and not minimal_loss_objective: # 圧縮優先の場合、圧縮損失を重視した損失を再計算
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

                            # 勾配復帰の強さ。
                            # forward値は変えず、backwardだけproxy側へ流す。
                            proxy_grad_weight = float(
                                getattr(args, "compression_primary_proxy_grad_weight", 0.10)
                            )

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
                    # build_compression_primary_loss が返した L は、
                    # 後から追加した gradient-only proxy をまだ含んでいない。

                    # ============================================================
                    # Actual Occupancy hard統計 + soft proxy勾配のSTE項
                    # ============================================================
                    # forward値は hard_octree_occupancy_stats と同じActual値にする。
                    # backwardは既存のsoft圧縮proxy/Actuator soft termsへ流す。
                    # これにより、Predicted Occupancyの数値はActual定義に揃えつつ、
                    # 学習時にはNetwork側へ勾配を返す。
                    # ============================================================
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

                    # ============================================================
                    # Prune Where direct gradient anchor
                    # ============================================================
                    # L_com_objective.requires_grad=True でも、勾配がMoveにしか流れていない場合がある。
                    # そのため、requires_gradの有無ではなく、Prune Where専用proxyを常に探して、
                    # forward値0のgradient-only項としてL_com_objectiveへ追加する。
                    #
                    # target_drop_ratioへ寄せるMSEは使わない。
                    # 目的は drop_head の勾配0を防ぐことだけである。
                    # ============================================================
                    prune_where_direct_weight = float(
                        getattr(args, "compression_soft_prune_logit_direct_grad_weight", 0.01)
                    )

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
                elif _discrete_loss_mode_value(args) == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None) # モデルが保持しているHard離散方策用の損失関数を取得する
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy
                base_model_for_correction = model.module if hasattr(model, "module") else model
                full_cloud_correction_loss, full_cloud_correction_debug = build_full_cloud_actual_correction_loss(
                    args=args,
                    correction_state=full_cloud_correction_state,
                    actuator_voxel_state=getattr(base_model_for_correction, "last_actuator_voxel_state", None),
                    reference=gen_xyz if torch.is_tensor(gen_xyz) else input_xyz,
                    global_step=global_train_step,
                )
                # ============================================================
                # Phase3:
                # full-cloud actual correction を compression terms にも保存する。
                # これにより step_grad / debug / compression_primary 側で追跡できる。
                # ============================================================
                if torch.is_tensor(full_cloud_correction_loss):
                    try:
                        phase3_terms = dict(getattr(loss, "last_compression_terms", {}) or {})
                        phase3_terms["full_cloud_actual_correction"] = full_cloud_correction_loss
                        loss.last_compression_terms = phase3_terms
                    except Exception:
                        pass

                if bool(getattr(args, "full_cloud_actual_correction_loss_enable", False)):
                    # ============================================================
                    # FullCloud anchor shadow training:
                    # full_cloud_actual_correction_loss は build_compression_primary_loss()
                    # より後で作られるため、compression_primary_modeでもここでLへ足す。
                    # forward値はFullCloud悪化時のsoft操作ペナルティであり、
                    # backwardはshadow SubtreeのActuator soft量へ流れる。
                    # ============================================================
                    correction_weight = (
                        float(getattr(args, "cp_full_cloud_actual_correction_weight", 0.05))
                        if bool(compression_primary_mode)
                        else float(getattr(args, "full_cloud_actual_correction_weight", 0.05))
                    )
                    add_correction_directly = correction_weight > 0.0

                    correction_block = correction_weight * full_cloud_correction_loss
                    correction_scale = 1.0
                    correction_balance = None

                    if add_correction_directly and bool(compression_primary_mode):
                        correction_balance = _compression_primary_support_balance(
                            args,
                            compression_support_anchor if torch.is_tensor(compression_support_anchor) else L,
                            correction_block,
                            enabled=uses_actual_total_bit_objective(args),
                            target_ratio_name="compression_primary_tail_target_ratio",
                            min_scale_name="compression_primary_tail_balance_min_scale",
                            max_scale_name="compression_primary_tail_balance_max_scale",
                            disabled_reason="tail_balance_disabled",
                        )
                        correction_scale = float(correction_balance["scale"])
                        L = L + correction_scale * correction_block
                    elif add_correction_directly:
                        L = L + correction_block

                    if isinstance(full_cloud_correction_debug, dict):
                        full_cloud_correction_debug["full_cloud_corr_loss_added_to_total"] = bool(add_correction_directly)
                        full_cloud_correction_debug["full_cloud_corr_loss_weight_used"] = float(correction_weight)
                        full_cloud_correction_debug["full_cloud_corr_loss_scale_to_total"] = float(correction_scale)
                        full_cloud_correction_debug["full_cloud_corr_loss_added_via_compression_primary"] = bool(
                            compression_primary_mode and add_correction_directly
                        )
                        full_cloud_correction_debug["full_cloud_corr_loss_requires_grad"] = bool(
                            torch.is_tensor(full_cloud_correction_loss)
                            and full_cloud_correction_loss.requires_grad
                        )
                    if isinstance(cp_debug, dict):
                        cp_debug["cp_support_correction_raw"] = case_float(correction_block, float("nan"))
                        cp_debug["cp_support_correction_scaled"] = case_float(
                            correction_scale * correction_block,
                            float("nan"),
                        )
                        cp_debug["cp_support_correction_scale"] = float(correction_scale)
                        cp_debug["cp_support_correction_reason"] = str(
                            (correction_balance or {}).get("reason", "")
                        )
                        cp_debug["cp_support_correction_target_ratio"] = (
                            float(correction_balance["target_ratio"])
                            if correction_balance is not None
                            and correction_balance.get("target_ratio", None) is not None
                            else float("nan")
                        )
                        cp_debug["cp_support_correction_dominant"] = str(
                            (correction_balance or {}).get("dominant", "neutral")
                        )
                        aux_scaled = float(cp_debug.get("cp_aux_block_scaled", 0.0))
                        tail_scaled = float(cp_debug.get("cp_support_tail_scaled", 0.0))
                        correction_scaled = case_float(correction_scale * correction_block, 0.0)
                        cp_debug["cp_support_total_scaled"] = float(
                            aux_scaled + tail_scaled + correction_scaled
                        )
                        main_block_value = float(cp_debug.get("cp_main_block", 0.0))
                        cp_debug["cp_support_total_ratio_to_main"] = (
                            abs(aux_scaled + tail_scaled + correction_scaled)
                            / max(abs(main_block_value), 1e-12)
                        )
                        cp_debug["cp_support_dominant"] = (
                            "compression"
                            if abs(main_block_value) + 1e-12 >= abs(aux_scaled + tail_scaled + correction_scaled)
                            else "support"
                        )
                else:
                    if isinstance(full_cloud_correction_debug, dict):
                        full_cloud_correction_debug["full_cloud_corr_loss_added_to_total"] = False
                        full_cloud_correction_debug["full_cloud_corr_loss_requires_grad"] = bool(
                            torch.is_tensor(full_cloud_correction_loss)
                            and full_cloud_correction_loss.requires_grad
                        )

                # ============================================================
                # Ablation: 圧縮損失のみで backward する
                # ============================================================
                # 目的:
                #   幾何損失 L_geom、属性損失 L_attr、方策損失 L_policy、
                #   操作損失 L_actuator、FullCloud補正損失などを
                #   optimizer更新に一切使わない。
                #
                # 注意:
                #   L_com_objective は train.py 側で構成された
                #   「学習用の圧縮目的」である。
                #   まずはこちらを使う方が、現在の圧縮学習経路を保ったまま
                #   他損失だけを除外できる。
                # ============================================================
                compression_only_ablation = True

                if compression_only_ablation:
                    L = L_com_objective
                    L_downstream = L
                    L_discrete_policy = L.new_zeros(())

                    if isinstance(cp_debug, dict):
                        cp_debug["compression_only_ablation"] = True
                        cp_debug["compression_only_ablation_loss"] = "L_com_objective"
                        
                """情報精査"""
                comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {}) # 直前の圧縮Debug情報を取り出す
                # Phase7-3: Network経路debugをcompression debugへ集約する。
                base_model_for_phase7 = model.module if hasattr(model, "module") else model
                phase7_structure_debug = getattr(base_model_for_phase7, "last_structure_debug", {}) or {}
                _phase7_update_from_structure(
                    comp_debug,
                    phase7_structure_debug,
                    is_anchor_step=bool(is_anchor_step and not full_cloud_anchor_shadow_train_active),
                )
                _phase7_update_from_voxel_state(comp_debug, model)
                # Phase7-4:
                # ablation modeと短時間判定用summaryをcomp_debugへ集約する。
                _phase7_add_ablation_summary_to_comp_debug(args, comp_debug)
                if isinstance(step_timing_breakdown, dict) and step_timing_breakdown:
                    comp_debug.update(step_timing_breakdown)
                    comp_debug["octree_build_time"] = float(
                        step_timing_breakdown.get("full_cloud_canonical_build_time", 0.0)
                        + step_timing_breakdown.get("subtree_group_build_time", 0.0)
                        + step_timing_breakdown.get("subtree_potential_select_time", 0.0)
                        + step_timing_breakdown.get("selected_metadata_oracle_time", 0.0)
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
                if isinstance(full_cloud_anchor_debug_snapshot, dict) and full_cloud_anchor_debug_snapshot:
                    comp_debug["full_cloud_anchor_shadow_train_requested"] = bool(full_cloud_anchor_shadow_train_requested)
                    comp_debug["full_cloud_anchor_shadow_train_used"] = bool(full_cloud_anchor_shadow_train_active)
                    comp_debug["full_cloud_anchor_optimizer_updates"] = bool(full_cloud_anchor_shadow_train_active)
                    comp_debug["full_cloud_anchor_actual_total_bit_percent"] = full_cloud_anchor_debug_snapshot.get(
                        "actual_total_bit_percent",
                        full_cloud_anchor_debug_snapshot.get("actual_bit_percent", None),
                    )
                    comp_debug["full_cloud_anchor_actual_bit_percent"] = full_cloud_anchor_debug_snapshot.get(
                        "actual_bit_percent",
                        full_cloud_anchor_debug_snapshot.get("actual_total_bit_percent", None),
                    )
                    comp_debug["full_cloud_anchor_teacher_type"] = full_cloud_anchor_debug_snapshot.get("teacher_type", "")
                    comp_debug["full_cloud_anchor_full_cloud_teacher_used"] = bool(
                        full_cloud_anchor_debug_snapshot.get("full_cloud_teacher_used", False)
                    )
                    comp_debug["full_cloud_anchor_point_count_before"] = full_cloud_anchor_debug_snapshot.get("point_count_before", None)
                    comp_debug["full_cloud_anchor_point_count_after"] = full_cloud_anchor_debug_snapshot.get("point_count_after", None)
                    comp_debug["full_cloud_anchor_unique_coord_before"] = full_cloud_anchor_debug_snapshot.get("unique_coord_before", None)
                    comp_debug["full_cloud_anchor_unique_coord_after"] = full_cloud_anchor_debug_snapshot.get("unique_coord_after", None)
                    for oracle_metric_key in (
                        "actual_train_objective_percent",
                        "policy_actual_percent",
                        "oracle_teacher_actual_percent",
                        "policy_full_cloud_actual_bit_percent",
                        "policy_final_full_cloud_raw_bit_percent",
                        "policy_final_full_cloud_actual_bit_percent",
                        "policy_final_full_cloud_gt_bit",
                        "policy_final_full_cloud_gen_bit",
                        "policy_final_full_cloud_total_bit_with_edit_record",
                        "oracle_full_cloud_raw_bit_percent",
                        "oracle_full_cloud_actual_bit_percent",
                        "oracle_full_cloud_override_used",
                    ):
                        if oracle_metric_key in full_cloud_anchor_debug_snapshot:
                            comp_debug[oracle_metric_key] = full_cloud_anchor_debug_snapshot.get(oracle_metric_key)
                    if full_cloud_primary_override_debug:
                        comp_debug.update(full_cloud_primary_override_debug)
                    if bool(comp_debug.get("full_cloud_actual_primary_used", False)):
                        subtree_actual_before = finite_float_or_none(
                            comp_debug.get("actual_total_bit_percent", comp_debug.get("total_bit", None))
                        )
                        full_actual_primary_value = finite_float_or_none(
                            comp_debug.get("full_cloud_actual_primary_forward_value", None)
                        )
                        if full_actual_primary_value is not None:
                            comp_debug["subtree_actual_bit_percent"] = (
                                float(subtree_actual_before) if subtree_actual_before is not None else None
                            )
                            comp_debug["subtree_teacher_percent"] = (
                                float(subtree_actual_before) if subtree_actual_before is not None else None
                            )
                            for anchor_key, output_key in (
                                ("gt_actual_bit", "gt_actual_bit"),
                                ("gen_actual_bit", "gen_actual_bit"),
                                ("gt_bit_abs", "gt_bit_abs"),
                                ("gen_bit_abs", "gen_bit_abs"),
                                ("actual_total_bits", "actual_total_bits"),
                                ("gen_total_bit_with_edit_record", "gen_total_bit_with_edit_record"),
                                ("actual_raw_percent", "actual_raw_percent"),
                                ("actual_edit_record_bits", "actual_edit_record_bits"),
                                ("gt_actual_encode_time", "gt_actual_encode_time"),
                                ("gen_actual_encode_time", "gen_actual_encode_time"),
                                ("actual_encode_time_total", "actual_encode_time_total"),
                                ("point_count_before", "point_count_before"),
                                ("point_count_after", "point_count_after"),
                                ("unique_coord_before", "unique_coord_before"),
                                ("unique_coord_after", "unique_coord_after"),
                            ):
                                if anchor_key in full_cloud_anchor_debug_snapshot:
                                    comp_debug[output_key] = full_cloud_anchor_debug_snapshot.get(anchor_key)
                            comp_debug["local_proxy_percent"] = comp_debug.get(
                                "proxy_total_bit_percent",
                                comp_debug.get("surrogate_total_bit_percent", None),
                            )
                            comp_debug["full_cloud_actual_bit_percent"] = float(full_actual_primary_value)
                            comp_debug["full_cloud_actual_percent"] = float(full_actual_primary_value)
                            comp_debug["actual_total_bit_percent"] = float(full_actual_primary_value)
                            comp_debug["actual_train_objective_percent"] = float(full_actual_primary_value)
                            comp_debug["actual_bit_percent"] = float(full_actual_primary_value)
                            comp_debug["actual_target"] = float(full_actual_primary_value)
                            comp_debug["actual_forward_value"] = float(full_actual_primary_value)
                            comp_debug["compression_forward_teacher_percent"] = float(full_actual_primary_value)
                            comp_debug["forward_display_value"] = float(full_actual_primary_value)
                            if bool(comp_debug.get("oracle_full_cloud_override_used", False)):
                                comp_debug["policy_action_source"] = "actual_oracle_full_cloud_override"
                                comp_debug["policy_actual_noop_guard_used"] = False
                                comp_debug["oracle_teacher_actual_percent"] = comp_debug.get(
                                    "oracle_full_cloud_actual_bit_percent",
                                    float(full_actual_primary_value),
                                )
                            else:
                                comp_debug["policy_actual_percent"] = float(full_actual_primary_value)
                            if subtree_actual_before is not None:
                                comp_debug["full_vs_subtree_actual_gap"] = float(full_actual_primary_value) - float(subtree_actual_before)
                                comp_debug["sign_match_subtree_full"] = bool(
                                    (float(full_actual_primary_value) <= 0.0 and float(subtree_actual_before) <= 0.0)
                                    or (float(full_actual_primary_value) >= 0.0 and float(subtree_actual_before) >= 0.0)
                                )
                            proxy_value_for_match = finite_float_or_none(comp_debug.get("local_proxy_percent", None))
                            if proxy_value_for_match is not None:
                                comp_debug["proxy_full_actual_gap"] = float(proxy_value_for_match) - float(full_actual_primary_value)
                                comp_debug["sign_match_proxy_full"] = bool(
                                    (float(full_actual_primary_value) <= 0.0 and float(proxy_value_for_match) <= 0.0)
                                    or (float(full_actual_primary_value) >= 0.0 and float(proxy_value_for_match) >= 0.0)
                                )
                            comp_debug["actual_value_source"] = "fresh_full_cloud_primary_ste"
                            comp_debug["actual_value_is_fresh"] = True
                            comp_debug["actual_scope"] = "full_cloud"
                            comp_debug["teacher_scope"] = "full_cloud_primary"
                            comp_debug["full_cloud_teacher_used"] = True
                    elif full_cloud_primary_override_debug:
                        comp_debug.update(full_cloud_primary_override_debug)

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


                if cp_debug: # Compression Primaryモード用のDebug情報が存在するか判定
                    comp_debug.update(cp_debug) # 圧縮目的のDebug情報を追加
                    loss.last_compression_debug = comp_debug # 統合後のcomp_debugをLossに保存
                if isinstance(full_context_subtree_delta_debug, dict) and full_context_subtree_delta_debug:
                    comp_debug.update(full_context_subtree_delta_debug)
                    loss.last_compression_debug = comp_debug
                if isinstance(last_full_cloud_correction_update_debug, dict) and last_full_cloud_correction_update_debug:
                    comp_debug.update(last_full_cloud_correction_update_debug)
                if isinstance(full_cloud_correction_debug, dict) and full_cloud_correction_debug:
                    comp_debug.update(full_cloud_correction_debug)
                    comp_debug["phase3_full_context_subtree_delta_in_terms"] = bool(
                        "full_context_subtree_delta" in terms
                    )
                    comp_debug["phase3_full_cloud_correction_in_terms"] = bool(
                        "full_cloud_actual_correction" in getattr(loss, "last_compression_terms", {})
                    )
                    comp_debug["phase3_compression_primary_mode"] = bool(compression_primary_mode)
                    comp_debug["phase3_full_context_requires_grad"] = bool(
                        torch.is_tensor(L_full_context_subtree_delta)
                        and L_full_context_subtree_delta.requires_grad
                    )
                    comp_debug["phase3_full_cloud_correction_requires_grad"] = bool(
                        torch.is_tensor(full_cloud_correction_loss)
                        and full_cloud_correction_loss.requires_grad
                    )

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

                if isinstance(full_cloud_correction_state, dict):
                    comp_debug.update(
                        {
                            "full_cloud_corr_ema_full_vs_subtree_gap": full_cloud_correction_state.get("ema_full_vs_subtree_gap"),
                            "full_cloud_corr_ema_full_vs_context_gap": full_cloud_correction_state.get("ema_full_vs_context_gap"),
                            "full_cloud_corr_ema_full_vs_proxy_gap": full_cloud_correction_state.get("ema_full_vs_proxy_gap"),
                            "full_cloud_corr_ema_full_actual_delta": full_cloud_correction_state.get("ema_full_actual_delta"),
                            "full_cloud_corr_last_full_actual_delta": full_cloud_correction_state.get("last_full_actual_delta"),
                            "full_cloud_corr_last_subtree_actual_delta": full_cloud_correction_state.get("last_subtree_actual_delta"),
                            "full_cloud_corr_last_full_context_delta": full_cloud_correction_state.get("last_full_context_delta"),
                            "full_cloud_corr_last_subtree_proxy_delta": full_cloud_correction_state.get("last_subtree_proxy_delta"),
                            "full_cloud_corr_last_update_step": full_cloud_correction_state.get("last_update_step"),
                        }
                    )

                comp_debug["full_cloud_corr_loss_added_to_total"] = bool(
                    getattr(args, "full_cloud_actual_correction_loss_enable", False)
                )
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
                for debug_key in ( # 圧縮CSVからも構造入力モードを追えるように必要項目だけを転記する
                    "use_subtree_tree",
                    "use_full_octree_context",
                    "octree_input_mode",
                    "structural_voxel_mode",
                    "point_feature_voxel_mode",
                    "structural_voxel_key_available",
                    "point_feature_voxel_key_available",
                    "selected_subtree_key",
                    "selected_subtree_path",
                    "root_to_subtree_path",
                    "global_offset",
                    "local_offset",
                    "global_depth",
                    "local_depth",
                    "parent_occupancy_code",
                    "sibling_count",
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
                    skip_optimizer_reason = None

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
                        and not bool(full_cloud_anchor_shadow_train_active)
                    ):
                        skip_optimizer_reason = "full_cloud_anchor_no_grad"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        loss.last_compression_debug = comp_debug

                    elif ( bool(getattr(args, "skip_optimizer_on_actual_fallback", True)) and bool(comp_debug.get("actual_codec_fallback_to_proxy", False))):
                        skip_optimizer_reason = "actual_codec_fallback_to_proxy"
                        comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                        loss.last_compression_debug = comp_debug

                """CSV"""
                compression_metric_row = build_compression_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, L_com=L_com) # 圧縮StepCSVに書き込む1行を作る
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
                        "full_cloud_anchor_shadow_train_requested",
                        "full_cloud_anchor_shadow_train_used",
                        "full_cloud_anchor_optimizer_updates",
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
                operation_metric_row = build_operation_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats) # 点操作StepCSVに書き込む1行を作る
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
                    ("full_context_subtree_delta", L_full_context_subtree_delta),
                    ("full_context_subtree_delta", L_full_context_subtree_delta),
                    ("full_cloud_actual_correction", full_cloud_correction_loss),
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
                ]
                if torch.is_tensor(La_fit) and La_fit.requires_grad:
                    step_grad_loss_items.append(("La_fit", La_fit))
                sparsepcgc_aux_term = terms.get("sparsepcgc", None)
                if torch.is_tensor(sparsepcgc_aux_term) and sparsepcgc_aux_term.requires_grad:
                    step_grad_loss_items.append(("sparsepcgc_aux_objective", sparsepcgc_aux_term))
                if (
                    bool(is_anchor_step)
                    and bool(full_cloud_anchor_no_grad)
                    and not bool(full_cloud_anchor_shadow_train_active)
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
                amp_info = { "enabled": bool(amp_scaler_enabled), "found_inf": None, "scale_before": None, "scale_after": None, "consecutive_amp_skips": int(consecutive_amp_skips)} # AMPの状態を記録する辞書を作る
                last_nonfinite_grad_summary = None
                if total_loss_finite: # 総損失がInfでないとき、更新前パラメータを記録
                    param_update_snapshots = capture_param_update_snapshots( args, model, step + 1, num_steps)
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

                    if bool(getattr(args, "debug_grad_flow", False)) or compact_step_text_log:
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
                maybe_record_case_debug( args, writer, case_debug_path, case_debug_counts, global_step=global_train_step, episode=episode, epoch=epoch, step=step, file_path=file_path, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats, L=L, L_geom=L_geom, L_com=L_com, L_actuator=L_actuator) # 圧縮改善が良いケース・悪いケースを条件に応じてCase Debag CSVへ保存

                """損失ログの記録"""
                if epoch_metric_sums is None:
                    epoch_metric_sums = new_metric_sums(L.device, plot.num_loss) # Epoch内で初めのStepなら損失累積器を作る
                surrogate_compression_metric = surrogate_compression_plot_metric(loss, L_com, L.device) # Surrogate予測の(Mine-GT)*100/GTを通常plotへ渡す
                actual_compression_metric = actual_compression_plot_metric(loss, L.device) # 実codecで測った(Mine-GT)*100/GTを通常plotへ渡す
                policy_actual_metric = policy_actual_compression_plot_metric(loss, L.device) # Network自身の最終出力actualを通常plotへ渡す
                oracle_teacher_metric = oracle_teacher_compression_plot_metric(loss, L.device) # Oracle teacher actualを通常plotへ渡す
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
                plot_edit_stats["oracle_full_cloud_prune_ratio_percent"] = operation_metric_row.get(
                    "oracle_full_cloud_prune_ratio_percent",
                    0.0,
                )
                plot.record_point_edits("step", global_train_step + 1, plot_edit_stats) # 点操作統計をCSVに記録
                plot.record_occupancy_metrics("step", global_train_step + 1, compression_metric_row) # 占有pattern/probability proxyと実hard octree統計をCSVに記録
                plot.record_voxel_collision_metrics("step", global_train_step + 1, compression_metric_row) # SparsePCGC量子化後の点潰れ率をCSV/plotへ記録
                plot_step_info = plot.record_metrics("step", global_train_step + 1, step_metric_values) # Step単位の損失値をCSVに保存
                if plot_step_info.get("skipped", False) and not compact_step_text_log:
                    threshold_text = f"{plot_step_info.get('threshold', float('nan')):.6g}"
                    baseline = plot_step_info.get("baseline", None)
                    baseline_text = ""
                    if baseline is not None:
                        baseline_text = f", baseline={float(baseline):.6g}"
                    writer.write( "PlotSkipStep: " f"global_step={global_train_step + 1}, " f"episode={episode + 1}, " f"epoch={epoch + 1}, " f"metric={plot_step_info.get('metric_key', 'unknown')}, " f"value={float(plot_step_info.get('value', float('nan'))):.6g}, " f"rule={plot_step_info.get('reason', 'unknown')}, " f"threshold={threshold_text}" f"{baseline_text}")
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
                point_count_min = min(subtree_point_counts) if subtree_point_counts else None
                point_count_max = max(subtree_point_counts) if subtree_point_counts else None
                point_count_mean = ( sum(subtree_point_counts) / float(len(subtree_point_counts)) if subtree_point_counts else None)
                subtree_meta_for_better = { "enabled": bool(subtree_mode), "depth": subtree_depth_meta.get("depth"), "base_depth": subtree_depth_meta.get("base_depth"), "min_depth": subtree_depth_meta.get("min_depth"), "max_depth": subtree_depth_meta.get("max_depth"), "uncapped_min_depth": subtree_depth_meta.get("uncapped_min_depth"), "uncapped_max_depth": subtree_depth_meta.get("uncapped_max_depth"), "data_max_depth": subtree_depth_meta.get("data_max_depth"), "curriculum_phase": subtree_depth_meta.get("curriculum_phase"), "percent_mode": subtree_depth_meta.get("depth_percent_curriculum"), "percent_range": subtree_depth_meta.get("depth_percent_range"), "point_count_min": point_count_min, "point_count_mean": point_count_mean, "point_count_max": point_count_max, "selected_subtree_count": selected_subtree_count, "eligible_subtree_count": eligible_subtree_count, "actual_eligible_subtree_count": actual_eligible_subtree_count, "total_subtree_count": total_subtree_count, "min_subtree_points": min_subtree_points, "is_anchor_step": bool(is_anchor_step), "anchor_reason": anchor_reason, "loss_scope": subtree_loss_scope, "subset_step": bool(subset_step), "subset_enabled": bool(subset_enabled)}
                log_for_better_step( for_better_path, args=args, model=model, loss_obj=loss, optimizer=optimizer, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, stage_factors=stage_factors, compression_row=compression_metric_row, operation_row=operation_metric_row, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats, subtree_meta=subtree_meta_for_better, loss_values={ "L": L, "L_geom": L_geom, "L_com": L_com, "L_com_objective": L_com_objective, "L_attr": L_attr, "L_policy": L_policy, "L_actuator": L_actuator, "loss_bit": loss_bit, "loss_single": loss_single, "loss_nodes": loss_nodes}, step_completed=step_completed, total_loss_finite=total_loss_finite, amp_info=amp_info, timing={"step_seconds": en_step - st_step})
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
            plot.record_occupancy_metrics("epo", global_epoch + 1) # Epoch内で記録された占有pattern/probability統計を集計
            plot.record_voxel_collision_metrics("epo", global_epoch + 1) # Epoch内のSparsePCGC量子化点潰れ率を集計
            log_epoch_point_edit_average( writer, epoch_edit_info, global_epoch) # Epoch単位の点ん操作統計をログに記録
            global_epoch += 1
            plot.plot_loss_curve("step")
            plot.plot_loss_curve("epo")
            plot.plot_point_edit_curve("step")
            plot.plot_point_edit_curve("epo")
            plot.plot_occupancy_curve("step")
            plot.plot_occupancy_curve("epo")
            plot.plot_voxel_collision_curve("step")
            plot.plot_voxel_collision_curve("epo")
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
        plot.plot_loss_curve("epi")
        plot.plot_point_edit_curve("epi")
        plot.plot_occupancy_curve("epi")
        plot.plot_voxel_collision_curve("epi")
        writer.write(f"Saved episode plots/csv: {plot.save_dir}")
        writer.flush()
        checkpoint_metrics = finalize_checkpoint_metrics( args, current_stage, episode, plot, episode_checkpoint_sums, checkpoint_gate_refs)
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
        operation_episode_metrics = finalize_operation_episode_metrics( episode, current_stage, episode_operation_sums)
        append_csv_row( metric_csv_paths.get("operation_episode"), OPERATION_EPISODE_METRIC_COLUMNS, operation_episode_metrics)

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
        torch.backends.cudnn.benchmark = not bool(getattr(args, "deterministic", False))
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
        p.requires_grad = False
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
        writer.close()
