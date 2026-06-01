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
from models.utils.pointcloud.quant_noise import add_uniform_quantization_noise, resolve_uniform_noise_delta
from models.utils.pointcloud.voxel_collision import (
    compute_voxel_collision_stats_batch,
    flatten_voxel_collision_stats,
    format_voxel_collision_summary,
)
from models.utils.data.dataset import *
from models.utils.patching.patch import *
from models.utils.compression.octree_stats import hard_octree_occupancy_stats
from models.utils.training.utils_grad import *
from models.utils.config.args import parse_pugan_args

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
from models.utils.training.case_debug import *
from models.utils.training.metric_csv import *
from models.utils.training.metric_columns import LOSS_GRAD_PROBE_COLUMNS
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
            "actuator.move_amount_head.",
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

        # どのくらい動かすか：移動割合を出すhead
        ("move_amount_head", [
            "actuator.move_amount_head.",
        ]),

        # ============================================================
        # 注意：現在のActuatorには「どの点をMove元にするか」専用headはない
        # move source選択は policy_probs / cause_scores / repair_gate / move_score から作られる。
        # そのため、source位置選択への勾配は policy_module や cost_attributor 側にも分散する。
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
            if hasattr(writer, "write"):
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
        if hasattr(writer, "write"):
            writer.write(format_voxel_collision_summary(stage, stats))
            note = str(stats.get("sampling_note", ""))
            if note:
                writer.write(f"VoxelCollisionSampling[{stage}]: {note}")
    return flat

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
    seq_dirs = collect_seq_dirs2(args.input_dir, dataset_name=args.dataname) # 入力ディレクトリから学習対象のシーケンスディレクトリ一覧を集める
    num_seq = len(seq_dirs)
    writer.write(f"Total seq directories: {num_seq}")
    seq_datasets = [(seq_dir, PlyDirDataset(args, seq_dir)) for seq_dir in seq_dirs] # 各シーケンス内のPLY点群ファイルを読み込むデータセットを作る
    total_train_files = sum(len(dataset) for _, dataset in seq_datasets) # 全シーケンスに含まれる点群ファイル数を合計し、総Step数の見積もりなどに使用
    args._total_train_steps_estimate = max(int(getattr(args, "episodes", 1)), 1) * max(int(total_train_files), 1) # Episode数と点群ファイル数からそう学修Step数を概算
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
                args._log_this_step = False
                sparsepcgc_csv_debug = ( str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc" and bool(getattr(args, "save_compression_metric_csv", True))) # Sparse PCGC専用ログ
                operation_csv_debug = bool( getattr(args, "save_operation_metric_csv", getattr(args, "save_operation_metrics_csv", True))) # 点操作メトリクスCSVを保存するか判定し、点移動量や追加/削除などのDebug収集条件に使用
                args._collect_sparsepcgc_debug = bool(sparsepcgc_csv_debug and should_collect_sparsepcgc_hard_debug(args, log_this_step=log_this_step, profile_this_step=profile_this_step, global_step=global_train_step)) # SparsePCGCの重いhard統計は毎Stepではなく診断間隔だけ収集する
                args._collect_structure_debug = bool( log_this_step or profile_this_step or operation_csv_debug or sparsepcgc_add_experiment_active(args))
                detail_log_this_step = False

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
                refresh_actual_gen = True # 実Codec/Surrogateの出力側更新を毎Step許可する

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
                    subtree_group_state = build_octree_subtree_groups_with_retry(
                        input_xyz,
                        args,
                        requested_subtree_depth,
                        min_subtree_points,
                        allow_largest_fallback=True,
                    )
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
                    selected_subtree_keys = candidate_subtree_keys # 初期状態では候補Subtreeを全て選択対象にする
                    if eligible_subtree_count > 0 and not is_anchor_step:
                        selected_subtree_keys = select_octree_subtree_keys(candidate_subtree_keys, global_train_step, args)
                        selected_subtree_keys = select_single_subtree_key( candidate_subtree_keys, selected_subtree_keys, global_train_step, args, cache_key) # 1StepでForwardするSubtreeをランダムに1個へ絞る
                    selected_subtree_count = int(selected_subtree_keys.numel()) # 実際に選択されたSubtree数を数える
                    subset_step = (not is_anchor_step) and selected_subtree_count < eligible_subtree_count # 候補の一部だけを使ったStepか否かの判定
                    encoder_debug_chunks = [] if detail_log_this_step else None # 詳細ログ対象Stepなら、各Subtree Forward時のEncoder Debugを保存するリスト

                    """Selected Groupsの作成"""
                    selected_groups = None
                    if not is_anchor_step: # Anchorでないとき
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
                    if not is_anchor_step:
                        t1 = time.time()
                        subtree_trees, full_octree_contexts, group_meta = build_selected_group_octree_metadata(
                            input_xyz,
                            subtree_ref,
                            selected_groups,
                            args=args,
                        )
                        t2 = time.time()
                        # print(t2-t1)
                    else:
                        subtree_trees = {}
                        full_octree_contexts = {}
                        group_meta = {}

                    """ログ"""
                    if log_this_step and bool(getattr(args, "train_patch_subset_log", True)):
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
                    gen_xyz = None
                    final_w = None
                    out_label = None

                    """モデルの実行"""
                    prev_log_flag = getattr(args, "_log_this_step", False)
                    try:
                        args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step) # このSubtree処理内で詳細ログを出すか否か決定
                        if is_anchor_step:
                            """全点群の場合"""
                            args._current_teacher_scope = "full_cloud" # full-cloud anchorでは実圧縮teacherも全点群基準として記録する
                            args._current_teacher_anchor_reason = str(anchor_reason) # full-cloudになった理由をteacherログへ渡す
                            args._current_exact_teacher_mode = "full_cloud" # exact occupancy teacherは全点群基準で走らせる
                            args._current_exact_teacher_uses_full_context = False # 全点群はSubtree文脈を使わない
                            args._current_exact_teacher_fallback_reason = "" # full-cloudではfallback理由なし
                            writer.write("Running full cloud Anchor step.") # Anchor Stepであることをログに出す
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # CUDAかつAMP有効なら混合精度計算の文脈を作る
                            with autocast_ctx: # 全体点群をモデルに入力し、出力点群と各種補助損失・点編集重みを得る
                                """モデルの実行"""
                                gen_pts, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward(
                                    input_xyz,
                                    input_attr_full,
                                    cache_key=cache_key,
                                    return_attr_output=False,
                                    subtree_ref=subtree_ref,
                                    selected_subtree_keys=None,
                                    subtree_tree=None,
                                    full_octree_context=None,
                                    octree_input_mode="full_cloud"
                                    )
                            if final_w is not None and not torch.isfinite(final_w).all(): # final重みにNanやinfが混ざっていないか確認
                                writer.write( "Warning: final_w contains NaN/Inf. " "It will be sanitized before point-edit summary and losses.")
                                final_w = torch.nan_to_num(final_w, nan=0.0, posinf=1.0, neginf=0.0) # 変換
                                final_w = final_w.clamp(0.0, 1.0) # 変換
                            if detail_log_this_step:
                                base_model = model.module if hasattr(model, "module") else model # DataParallelで包まれているばあいは中身のモデルを取り出す
                                encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {})) # Encoder Debug情報をコピーして保存
                            gen_xyz = gen_pts[:, :3, :]
                            train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 入力点群と出力点群を比較し、各操作の編集統計を計算
                            final_w_for_loss = None
                            if _discrete_loss_mode_value(args) != "hard":
                                final_w_for_loss = final_w
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext() # 形状損失と圧縮損失の計算もAMP文脈で行うための設定を作る
                            with autocast_ctx:
                                """形状損失の計算"""
                                L_geom = loss.get_geometry_loss( args, gen_pts=gen_xyz, gt_pts=input_xyz[:, :3, :], final_w=final_w_for_loss, out_label=out_label)

                                """圧縮損失の計算"""
                                if stage_factors["com"] != 0.0:
                                    compression_gen_xyz, noise_debug = prepare_compression_points( gen_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step)) # 圧縮損失用の入力点群を作る
                                    L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( # 圧縮損失の計算
                                        args,
                                        gen_xyz=compression_gen_xyz,
                                        gt_xyz=input_xyz[:, :3, :],
                                            final_w=final_w_for_loss,
                                            cache_key=cache_key,
                                            refresh_actual_gen=refresh_actual_gen,
                                            actual_gen_xyz=gen_xyz,
                                            subtree_tree=None,
                                            full_octree_context=None,
                                            octree_input_mode="full_cloud",
                                            )
                                else:
                                    writer.write("!!! Skipping compression loss calculation due to stage factor setting. !!!")
                                    zero = input_xyz.new_zeros(())
                                    L_com = zero
                                    loss_bit = zero
                                    loss_single = zero
                                    loss_nodes = zero
                        else:
                            """Subtreeの場合"""
                            writer.write("Running subtree step with selected Subtree.") # Subtree Stepであることをログに出す
                            num_selected = float(max(len(selected_groups), 1)) # 選択されたSubtree数をFloatで取得
                            subtree_edit_sums = new_point_edit_sums() # 複数Subtreeの点編集統計を累積するための変数を初期化
                            subtree_noise_debug_values = [] # 各Subtreeで圧縮用ノイズを加えたかなどを統合
                            subtree_compression_term_sums = {} # Subtreeごとの圧縮損失内訳を累積する辞書

                            for subtree_key, point_idx in selected_groups: # 選択されたSubtreeを1つずつ取り出し、それぞれ日いて点群を切り出し、Forward、形状損失、圧縮損失を計算
                                args._current_teacher_scope = "subtree_local" # Subtree stepではteacherが局所点群基準であることをLoss側へ渡す
                                args._current_subtree_id = str(subtree_key) # bad caseやteacherログでsubtree識別子を保存する
                                subtree_key_int = int(subtree_key) # 追加メタ辞書参照用にPython intへ揃える
                                subtree_tree = subtree_trees.get(subtree_key_int) # 事前構築Subtree Octreeを取得する
                                full_octree_context = full_octree_contexts.get(subtree_key_int) # full-cloud上のSubtree文脈を取得する
                                subtree_group_meta = group_meta.get(subtree_key_int, {}) or {} # ログ用メタ
                                use_subtree_tree = subtree_tree is not None # 構造評価に事前構築木を使えるか
                                use_full_octree_context = full_octree_context is not None # 大域文脈を使えるか
                                octree_input_mode = "prebuilt_subtree_tree" if use_subtree_tree else "local_recomputed" # 明示的な入力モード
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
                                if log_this_step:
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
                                        compression_subtree_xyz, noise_debug_sub = prepare_compression_points( gen_subtree_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step)) # Subtree出力を圧縮損失用に整える
                                        subtree_noise_debug_values.append(noise_debug_sub) # 現在Subtreeのノイズ情報を保持する
                                        L_com_sub, loss_bit_sub, loss_single_sub, loss_nodes_sub, _, _ = loss.get_compression_loss( # 損失計算
                                            args,
                                            gen_xyz=compression_subtree_xyz,
                                            gt_xyz=subtree_xyz[:, :3, :],
                                            final_w=final_w_sub_loss,
                                            cache_key=subtree_cache_key,
                                            refresh_actual_gen=refresh_actual_gen,
                                            actual_gen_xyz=gen_subtree_xyz,
                                            subtree_tree=subtree_tree,
                                            full_octree_context=full_octree_context,
                                            octree_input_mode=octree_input_mode,
                                            )
                                        accumulate_compression_terms( subtree_compression_term_sums, getattr(loss, "last_compression_terms", {}) or {}, 1.0 / num_selected) # 現在Subtreeで計算された圧縮損失内訳を1/Subtree数の重みをつけて、Step全体の圧縮損失内訳に累積する
                                    else:
                                        zero = subtree_xyz.new_zeros(())
                                        L_com_sub = zero
                                        loss_bit_sub = zero
                                        loss_single_sub = zero
                                        loss_nodes_sub = zero
                                

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
                terms = getattr(loss, "last_compression_terms", {}) or {} # 直前の圧縮損失の内訳の取得
                L_com_objective = compose_train_compression_objective(args, terms, L_com, La_fit) # actual/surrogateではL_com直結と内訳合成を半々で混ぜる
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
                if compression_primary_mode: # 圧縮優先の場合、圧縮損失を重視した損失を再計算
                    L, L_com_objective, cp_debug = build_compression_primary_loss(
                        args,
                        terms=terms,
                        L_com=L_com,
                        L_geom=L_geom,
                        L_actuator=L_actuator,
                        global_train_step=global_train_step,
                        stage_factors=stage_factors,
                    )

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
                                L_com_objective = L_com_objective + proxy_grad_weight * (
                                    compression_proxy_for_grad - compression_proxy_for_grad.detach()
                                )
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

                    L_downstream = L_com_objective

                    L = (
                        L
                        + stage_factors["attr"] * args.w_attr * L_attr
                        + stage_factors["policy"] * args.w_policy * L_policy
                        + stage_factors["repair"] * args.w_actuator * L_actuator
                    )
                    L_discrete_policy = L.new_zeros(())
                elif _discrete_loss_mode_value(args) == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None) # モデルが保持しているHard離散方策用の損失関数を取得する
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy

                """情報精査"""
                comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {}) # 直前の圧縮Debug情報を取り出す
                if cp_debug: # Compression Primaryモード用のDebug情報が存在するか判定
                    comp_debug.update(cp_debug) # 圧縮目的のDebug情報を追加
                    loss.last_compression_debug = comp_debug # 統合後のcomp_debugをLossに保存
                base_model = model.module if hasattr(model, "module") else model # DataParallelで包まれている場合は中身のモデルを取り出す
                structure_debug = getattr(base_model, "last_structure_debug", {}) or {} # モデル内部で記録された構造解析・構造修復のDebug情報を取得
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
                ):
                    if debug_key in structure_debug and debug_key not in comp_debug:
                        comp_debug[debug_key] = structure_debug.get(debug_key)
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
                    if ( log_this_step and bool(getattr(args, "surrogate_realign_on_low_corr", False)) and corr_value is not None and corr_value < float(getattr(args, "surrogate_realign_min_corr", 0.3))):
                        writer.write( "SurrogateRealignNotice: " f"corr_surrogate_actual={corr_value:.6f} below " f"{float(getattr(args, 'surrogate_realign_min_corr', 0.3)):.6f}; " f"realign_steps={int(getattr(args, 'surrogate_realign_steps', 0))} " "(current implementation logs the trigger; extra realign steps are not run unless added later).")
                skip_optimizer_reason = None
                if ( bool(getattr(args, "skip_optimizer_on_actual_fallback", True)) and bool(comp_debug.get("actual_codec_fallback_to_proxy", False))):
                    skip_optimizer_reason = "actual_codec_fallback_to_proxy"
                    comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                    loss.last_compression_debug = comp_debug

                """CSV"""
                compression_metric_row = build_compression_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, L_com=L_com) # 圧縮StepCSVに書き込む1行を作る
                operation_metric_row = build_operation_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats) # 点操作StepCSVに書き込む1行を作る

                """ログ"""
                if log_this_step:
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
                if total_loss_finite: # 総損失がInfでないとき、更新前パラメータを記録
                    param_update_snapshots = capture_param_update_snapshots( args, model, step + 1, num_steps)
                if skip_optimizer_reason is not None: # Optimizer更新を止める必要があるか否かの判定
                    writer.write( f"Skip Optimizing!!! reason={skip_optimizer_reason}; " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}") # Skip理由と位置を同じ行に出す
                    writer.write( "Skipped optimizer step because actual codec teacher fell back to proxy at " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}; " "this prevents proxy-only updates from replacing real-compression imitation.")
                elif not total_loss_finite:
                    writer.write( f"Skip Optimizing!!! reason=non_finite_total_loss; " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}, L={float(L.detach().float().mean().cpu()) if torch.is_tensor(L) else float('nan'):.6g}") # 非有限Lossの理由と値を同じ行に出す
                    writer.write( f"Skipped optimizer step due to non-finite total loss at " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}.")
                elif amp_scaler_enabled: # AMP用の逆伝播・更新処理へ進む
                    """AMP更新/勾配"""
                    scale_before = float(scaler.get_scale()) # BackWard前のAMP loss caleを取得
                    amp_info["scale_before"] = scale_before # AMP Debug情報に更新前ぉssSacleを保存
                    scaler.scale(L).backward() # LをAMP用にスケーリングしてから逆伝播
                    scaler.unscale_(optimizer) # Optimizer内の勾配を元のスケールへ戻す
                    if bool(getattr(args, "debug_grad_flow", False)):
                        log_grad_flow(args, writer, model, step + 1, num_steps, global_step=global_train_step) # 各層・各モジュールに勾配が届いているか否かの判定ログ
                    nonfinite_grad_summary = _summarize_nonfinite_grads(
                        model,
                        limit=int(getattr(args, "nonfinite_grad_log_param_limit", 8)),
                    )
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
                            torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip) # 学習対象パラメータの勾配ノルムをGrad Clip以下に制限
                        scaler.step(optimizer) # Optimizer更新
                        optimizer_state = scaler._per_optimizer_states[id(optimizer)] # このOptimizerに対するGradScaler内部状態を取得
                        found_inf = 0.0
                        if optimizer_state["found_inf_per_device"]:
                            found_inf = float( sum(v.item() for v in optimizer_state["found_inf_per_device"].values())) # GPUごとのInf検出値を合計し、、このStepでAMP Overflowが発生したかを数値化
                        scaler.update() # GradScalerのLoss Scaleを更新
                        scale_after = float(scaler.get_scale()) # 更新後Loss Scaleを取得
                        amp_info["found_inf"] = found_inf # Inf/NaN勾配の検出量をAMP Debug情報へ保存する
                        amp_info["scale_after"] = scale_after # 更新後Loss ScaleをAMP Debug情報へ保存
                        step_completed = found_inf == 0.0 and scale_after >= scale_before # Inf/NaNが検出されなければOptimizer更新成功とする
                        if step_completed: # 成功した場合の処理
                            consecutive_amp_skips = 0
                        else:
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
                    log_grad_flow(args, writer, model, step + 1, num_steps, global_step=global_train_step) # 各モジュールの勾配状態をログに出す
                    nonfinite_grad_summary = _summarize_nonfinite_grads(
                        model,
                        limit=int(getattr(args, "nonfinite_grad_log_param_limit", 8)),
                    )
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
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip) # 勾配爆発抑制
                        optimizer.step() # モデルパラメータの更新
                        step_completed = True # 更新フラグをTrueにする
                        consecutive_amp_skips = 0 # AMP loss scale連続Skip回数を0に戻す
                if step_completed: # Optimizer更新が成功したら差分ログを出す
                    log_param_updates( args, writer, model, param_update_snapshots, step + 1, num_steps)
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_step_end = time.time()
                epoch_has_optimizer_step = epoch_has_optimizer_step or step_completed # このEpoch内で一回でも更新が成功したかを記録
                if skip_optimizer_reason is not None or not total_loss_finite:
                    args._last_grad_flow = {} # backwardしていないskip stepでは前stepの勾配値をCSVへ持ち越さない
                operation_metric_row = attach_grad_flow_to_operation_row(operation_metric_row, args) # backward後に得られた各操作headの勾配normをOperation CSV行へ反映する
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
                surrogate_metrics = surrogate_plot_metrics(loss) # Surrogate教師学習の誤差系列を通常plotへ渡す
                metric_values = [ L, L_geom, surrogate_compression_metric, actual_compression_metric, L_attr, L_policy, loss_single, loss_nodes, Lp_out, La_fit, La_rep, L_actuator, *surrogate_metrics] # plot列順にStep損失をまとめる
                add_metric_sums( epoch_metric_sums, metric_values, L.device) # 現在Stepの損失値をEpoch累積器へ加算
                if episode_metric_sums is None:
                    episode_metric_sums = new_metric_sums(L.device, plot.num_loss) # Episode内で初めのEpochなら損失累積器を作る
                step_metric_values = metric_values # Step/Episode/Checkpointで同じ列順のmetricを使う
                add_metric_sums(episode_metric_sums, step_metric_values, L.device) # 現在Stepの損失一覧
                accumulate_checkpoint_metrics( episode_checkpoint_sums, compression_metric_row, operation_metric_row, step_metric_values) # ChackPoint判定用メトリクス
                if train_edit_stats is None:
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args) # 点操作情報を計算
                plot.record_point_edits("step", global_train_step + 1, train_edit_stats) # 点操作統計をCSVに記録
                plot.record_occupancy_metrics("step", global_train_step + 1, compression_metric_row) # 占有pattern/probability proxyと実hard octree統計をCSVに記録
                plot.record_voxel_collision_metrics("step", global_train_step + 1, compression_metric_row) # SparsePCGC量子化後の点潰れ率をCSV/plotへ記録
                plot_step_info = plot.record_metrics("step", global_train_step + 1, step_metric_values) # Step単位の損失値をCSVに保存
                if plot_step_info.get("skipped", False):
                    threshold_text = f"{plot_step_info.get('threshold', float('nan')):.6g}"
                    baseline = plot_step_info.get("baseline", None)
                    baseline_text = ""
                    if baseline is not None:
                        baseline_text = f", baseline={float(baseline):.6g}"
                    writer.write( "PlotSkipStep: " f"global_step={global_train_step + 1}, " f"episode={episode + 1}, " f"epoch={epoch + 1}, " f"metric={plot_step_info.get('metric_key', 'unknown')}, " f"value={float(plot_step_info.get('value', float('nan'))):.6g}, " f"rule={plot_step_info.get('reason', 'unknown')}, " f"threshold={threshold_text}" f"{baseline_text}")
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    en_step = time.time()

                    log_step_timing( writer=writer, args=args, step=step, num_steps=num_steps, epoch=epoch, global_train_step=global_train_step, use_cuda=use_cuda, st_step=st_step, timing_data_start=timing_data_start, timing_data_end=timing_data_end, timing_model_start=timing_model_start, timing_model_end=timing_model_end, timing_noise_start=timing_noise_start, timing_noise_end=timing_noise_end, timing_loss_start=timing_loss_start, timing_loss_end=timing_loss_end, timing_step_end=timing_step_end, en_step=en_step, loss=loss, model=model, KNN_BACKEND=KNN_BACKEND)
                else:
                    en_step = time.time()
                if log_this_step:
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
        append_csv_row( metric_csv_paths.get("checkpoint_episode"), CHECKPOINT_METRIC_COLUMNS, checkpoint_metrics)
        compression_episode_metrics = finalize_compression_episode_metrics( episode, current_stage, episode_compression_sums)
        append_csv_row( metric_csv_paths.get("compression_episode"), COMPRESSION_EPISODE_METRIC_COLUMNS, compression_episode_metrics)
        operation_episode_metrics = finalize_operation_episode_metrics( episode, current_stage, episode_operation_sums)
        append_csv_row( metric_csv_paths.get("operation_episode"), OPERATION_EPISODE_METRIC_COLUMNS, operation_episode_metrics)

        # 毎エピソードと最高スコアのモデルを保存
        best_loss, model_path, best_trackers = save_episode_checkpoint( model=model, ckpt_dir=ckpt_dir, plot=plot, writer=writer, episode=episode, best_loss=best_loss, args=args, stage=current_stage, checkpoint_metrics=checkpoint_metrics, best_trackers=best_trackers, loss=loss)
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
