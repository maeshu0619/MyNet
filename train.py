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
from models.utils.loss.loss import Loss
from models.utils.notify.mail_notify import TrainingMailNotifier
from record.write import Writing
from record.plot import PlotMaker
from models.utils.pointcloud.utils_repkpu import *
from models.utils.pointcloud.octree_subtree import *
from models.utils.pointcloud.quant_noise import add_uniform_quantization_noise, resolve_uniform_noise_delta
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
from models.utils.training.actual_codec_status import *
from models.utils.training.metric_rows import *
from models.utils.training.episode_metrics import *
from models.utils.training.checkpoint_metrics import *
from models.utils.training.actual_compression_guard import apply_actual_compression_guard
from models.utils.training.for_better_logging import *

from models.utils.surrogate.pretrain import *

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
    for episode in range(args.episodes): # Episode開始
        writer.write(f"◆◆◆ Episode {episode + 1} / {args.episodes} ◆◆◆")
        
        """Stage変更"""
        current_stage = resolve_training_stage_for_episode(args, episode) # 現在のEpisode番号から学習ステージを決め、形状重視、圧縮重視、Joint学習重視などの損失構成を切り替える
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
            loader = torch.utils.data.DataLoader(dataset, **loader_kwargs) # 現在のDatasetから点群ファイルを順に読み出す
            num_steps = len(dataset)
            epoch_has_optimizer_step = False
            epoch_metric_sums = None

            for step, pts in enumerate(loader): # Step開始
                """基本情報のセットアップ"""
                st_step = time.time()
                file_path = dataset.files[step]
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
                args._log_this_step = False
                sparsepcgc_csv_debug = ( str(getattr(args, "compress", "")).strip().lower().replace("-", "").replace("_", "") == "sparsepcgc" and bool(getattr(args, "save_compression_metric_csv", True))) # Sparse PCGC専用ログ
                operation_csv_debug = bool( getattr(args, "save_operation_metric_csv", getattr(args, "save_operation_metrics_csv", True))) # 点操作メトリクスCSVを保存するか判定し、点移動量や追加/削除などのDebug収集条件に使用
                args._collect_sparsepcgc_debug = bool(log_this_step or profile_this_step or sparsepcgc_csv_debug)
                args._collect_structure_debug = bool( log_this_step or profile_this_step or operation_csv_debug or sparsepcgc_add_experiment_active(args))
                detail_log_this_step = False
                
                """学習設定"""
                if timing_enabled and use_cuda and torch.cuda.is_available(): # GPU計測のためのリセット
                    torch.cuda.reset_peak_memory_stats()

                if timing_enabled: # 時間計測が有効なら入力整形処理の開始時刻を記録
                    sync_for_timing(use_cuda) # GPUを使用している場合は、正確な時間計測のためにGPUの処理が完了するのを待つ
                    timing_data_start = time.time() # 時間計測開始
                if subtree_mode: # Subtree部分学習モード
                    input_pcd = pts if pts.dim() == 3 else pts.unsqueeze(0) # 形式変換
                    input_pcd = downsample_input_batch(input_pcd, args, cache_key) # 点数が多すぎる場合に入力点群を間引き、GPUメモリ消費と計算時間を抑える
                    if use_cuda:
                        input_pcd = input_pcd.cuda(non_blocking=True) # 点群テンソルをGPUへ転送
                    input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous() # 形式変換
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
                # compute_compression = True if compression_primary_mode else stage_factors["com"] != 0.0 # このStepで圧縮損失を実際に計算するか決める
                compute_compression = True
                # refresh_actual_gen = not bool(getattr(args, "disable_actual_codec_during_train", False)) # 学習中に出力点群を実Codecに通して実圧縮結果を更新するか決める
                refresh_actual_gen = True

                """変数の初期化と設定"""
                subset_step = False # 部分学習か否か
                subset_enabled = False # 部分集合学習が有効か否か
                is_anchor_step = True # 初期状態では全体点ん群を使うAnchor Stepとする
                compression_cache_key = cache_key # キャッシュキーの初期化
                compression_gt_pts = input_xyz # 圧縮損失で比較する教師側点群を入力点群にする
                train_edit_stats = None # 点操作を見計算状態にする
                noise_debug = empty_noise_debug() # 圧縮損失用に量子化前の点群に加えるノイズのデバッグ情報を初期化
                subtree_depth_meta = {} # 深度などの情報保存
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
                    subtree_depth_meta = sample_train_subtree_depth( input_xyz, args, global_step=global_train_step, cache_key=cache_key) # Octree深度の決定
                    requested_subtree_depth = int(subtree_depth_meta["depth"]) # 深度を整数で取り出す
                    min_subtree_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1) # Subtreeとして採用する点数の最小点数
                    subtree_group_state = build_octree_subtree_groups_with_retry( input_xyz, args, requested_subtree_depth, min_subtree_points, allow_largest_fallback=True) # 入力点群から指定深度のOctree Subtree群を作る
                    
                    """Subtree情報"""
                    subtree_ref = subtree_group_state["subtree_ref"] # Subtree参照情報の抽出
                    if subtree_ref is None:
                        raise RuntimeError("Subtree mode did not find any valid octree subtree.")
                    subtree_depth_meta = dict(subtree_depth_meta) # 深度メタ情報変換
                    subtree_depth_meta["requested_depth"] = int(requested_subtree_depth) # 保存
                    subtree_depth_meta["depth"] = int(subtree_group_state.get("depth", requested_subtree_depth)) # 
                    subtree_depth_meta["retry_count"] = int(subtree_group_state.get("retry_count", 0))
                    subtree_depth_meta["selection_reason"] = str(subtree_group_state.get("selection_reason", "none"))
                    all_subtree_keys = subtree_group_state["unique_keys"]
                    subtree_index_lists = subtree_group_state["index_lists"]
                    all_groups = subtree_group_state["all_groups"]
                    total_subtree_count = int(all_subtree_keys.numel())
                    eligible_groups = list(subtree_group_state.get("eligible_groups", []))
                    actual_eligible_subtree_count = int(len(eligible_groups))
                    min_points_miss = bool(total_subtree_count > 0 and not eligible_groups and min_subtree_points > 1)
                    candidate_groups = eligible_groups or list(subtree_group_state.get("groups", [])) or all_groups
                    candidate_subtree_keys = all_subtree_keys.new_tensor(
                        [subtree_key for subtree_key, _ in candidate_groups],
                        dtype=all_subtree_keys.dtype,
                    ) if candidate_groups else all_subtree_keys.new_empty((0,), dtype=all_subtree_keys.dtype)
                    eligible_subtree_count = int(candidate_subtree_keys.numel())
                    is_anchor_step, anchor_reason = should_use_full_cloud_anchor( args, global_step=global_train_step, cache_key=cache_key)

                    if ( min_points_miss and eligible_subtree_count <= 0 and bool(getattr(args, "train_subtree_anchor_on_min_points_miss", False))):
                        is_anchor_step = True
                        anchor_reason = "min_points_miss_full_anchor"
                        log_for_better_event( for_better_path, "subtree_min_points_miss", global_step=global_train_step + 1, sampled_depth=int(subtree_depth_meta["depth"]), min_subtree_points=min_subtree_points, total_subtree_count=total_subtree_count, action="full_anchor")
                    elif min_points_miss:
                        log_for_better_event( for_better_path, "subtree_min_points_miss", global_step=global_train_step + 1, sampled_depth=int(subtree_depth_meta["depth"]), min_subtree_points=min_subtree_points, total_subtree_count=total_subtree_count, action="legacy_all_subtrees_fallback")

                    selected_subtree_keys = candidate_subtree_keys
                    if eligible_subtree_count > 0 and not is_anchor_step:
                        selected_subtree_keys = select_octree_subtree_keys(candidate_subtree_keys, global_train_step, args)
                    selected_subtree_count = int(selected_subtree_keys.numel())
                    subset_step = (not is_anchor_step) and selected_subtree_count < eligible_subtree_count
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    selected_groups = None
                    if not is_anchor_step:
                        selected_key_set = set(selected_subtree_keys.detach().cpu().tolist())
                        group_source = candidate_groups
                        selected_groups = [ (subtree_key, point_idx) for subtree_key, point_idx in group_source if subtree_key in selected_key_set]
                        if not selected_groups and group_source:
                            selected_groups = [max(group_source, key=lambda item: int(item[1].numel()))]
                        if not selected_groups:
                            raise RuntimeError("Subtree mode did not select any subtree group.")
                    if is_anchor_step:
                        subtree_point_counts = [int(point_idx.numel()) for _, point_idx in (eligible_groups or [])]
                        if not subtree_point_counts:
                            subtree_point_counts = [int(input_xyz.shape[-1])]
                        subtree_loss_scope = "full_cloud_output_vs_full_cloud_input"
                    else:
                        subtree_point_counts = [int(point_idx.numel()) for _, point_idx in selected_groups]
                        subtree_loss_scope = "subtree_output_vs_subtree_input"
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
                    prev_log_flag = getattr(args, "_log_this_step", False)
                    try:
                        args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step)
                        if is_anchor_step:
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                            with autocast_ctx:
                                ( gen_pts, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label) = model.forward( input_xyz, input_attr_full, cache_key=cache_key, return_attr_output=False, subtree_ref=subtree_ref, selected_subtree_keys=None)
                            if final_w is not None and not torch.isfinite(final_w).all():
                                writer.write( "Warning: final_w contains NaN/Inf. " "It will be sanitized before point-edit summary and losses.")
                                final_w = torch.nan_to_num(final_w, nan=0.0, posinf=1.0, neginf=0.0)
                                final_w = final_w.clamp(0.0, 1.0)
                            if detail_log_this_step:
                                base_model = model.module if hasattr(model, "module") else model
                                encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
                            gen_xyz = gen_pts[:, :3, :]
                            train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)
                            final_w_for_loss = None
                            if str(getattr(args, "discretelossmode", "hard")).strip().lower() != "hard":
                                final_w_for_loss = final_w
                            autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                            with autocast_ctx:
                                L_geom = loss.get_geometry_loss( args, gen_pts=gen_xyz, gt_pts=input_xyz[:, :3, :], final_w=final_w_for_loss, out_label=out_label)
                                if stage_factors["com"] != 0.0:
                                    compression_gen_xyz, noise_debug = prepare_compression_points( gen_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step))
                                    L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( args, gen_xyz=compression_gen_xyz, gt_xyz=input_xyz[:, :3, :], final_w=final_w_for_loss, cache_key=cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_xyz)
                                else:
                                    zero = input_xyz.new_zeros(())
                                    L_com = zero
                                    loss_bit = zero
                                    loss_single = zero
                                    loss_nodes = zero
                        else:
                            num_selected = float(max(len(selected_groups), 1))
                            subtree_edit_sums = new_point_edit_sums()
                            subtree_noise_debug_values = []
                            subtree_compression_term_sums = {}
                            for subtree_key, point_idx in selected_groups:
                                subtree_xyz = input_xyz.index_select(2, point_idx).contiguous()
                                subtree_attr = None
                                if input_attr_full is not None:
                                    subtree_attr = input_attr_full.index_select(2, point_idx).contiguous()
                                subtree_cache_key = ( f"{cache_key}|subtree_depth={int(subtree_ref['depth'][0].item())}|subtree_key={subtree_key}")
                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                                with autocast_ctx:
                                    ( gen_subtree_pts, L_attr_sub, L_policy_sub, L_actuator_sub, final_w_sub, Lp_out_sub, La_fit_sub, La_rep_sub, out_label_sub) = model.forward( subtree_xyz, subtree_attr, cache_key=subtree_cache_key, return_attr_output=False)
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                                gen_subtree_xyz = gen_subtree_pts[:, :3, :]
                                subtree_edit_stats = summarize_point_edits( input_xyz=subtree_xyz[:, :3, :], gen_pts=gen_subtree_pts, final_w=final_w_sub, args=args)
                                add_point_edit_sums(subtree_edit_sums, subtree_edit_stats)
                                final_w_sub_loss = None
                                if str(getattr(args, "discretelossmode", "hard")).strip().lower() != "hard":
                                    final_w_sub_loss = final_w_sub

                                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                                with autocast_ctx:
                                    L_geom_sub = loss.get_geometry_loss( args, gen_pts=gen_subtree_xyz, gt_pts=subtree_xyz[:, :3, :], final_w=final_w_sub_loss, out_label=out_label_sub)
                                    if stage_factors["com"] != 0.0:
                                        compression_subtree_xyz, noise_debug_sub = prepare_compression_points( gen_subtree_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step))
                                        subtree_noise_debug_values.append(noise_debug_sub)
                                        L_com_sub, loss_bit_sub, loss_single_sub, loss_nodes_sub, _, _ = loss.get_compression_loss( args, gen_xyz=compression_subtree_xyz, gt_xyz=subtree_xyz[:, :3, :], final_w=final_w_sub_loss, cache_key=subtree_cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_subtree_xyz)
                                        accumulate_compression_terms( subtree_compression_term_sums, getattr(loss, "last_compression_terms", {}) or {}, 1.0 / num_selected)
                                    else:
                                        zero = subtree_xyz.new_zeros(())
                                        L_com_sub = zero
                                        loss_bit_sub = zero
                                        loss_single_sub = zero
                                        loss_nodes_sub = zero

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
                elif args.split2patch:
                    optimizer.zero_grad(set_to_none=True)
                    patch_info = get_patch_info(input_pcd, args, cache_key, patch_info_cache)
                    total_patch_count = int(patch_info["num_patches"])
                    subset_enabled = bool(getattr(args, "train_patch_subset_enable", False))
                    selected_patch_ids = torch.arange( total_patch_count, device=patch_info["patch_xyz"].device, dtype=torch.long)
                    if subset_enabled:
                        is_anchor_step, _ = should_use_full_cloud_anchor( args, global_step=global_train_step, cache_key=cache_key)
                        if not is_anchor_step:
                            selected_patch_ids = select_patch_subset_ids(patch_info, global_train_step, args)
                    selected_patch_count = int(selected_patch_ids.numel())
                    subset_step = bool( subset_enabled and (not is_anchor_step) and selected_patch_count < total_patch_count)
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    pb = effective_patch_batch_size( args, patch_count=selected_patch_count, patch_size=args.num_points, is_train=True, writer=writer)
                    patch_outputs = []
                    patch_count = selected_patch_count
                    geom_weight_sum = 0.0
                    L_geom = input_pcd.new_zeros(())
                    L_attr = input_pcd.new_zeros(())
                    L_policy = input_pcd.new_zeros(())
                    L_actuator = input_pcd.new_zeros(())
                    Lp_out = input_pcd.new_zeros(())
                    La_fit = input_pcd.new_zeros(())
                    La_rep = input_pcd.new_zeros(())
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        prev_patch_geom_log = getattr(args, "_log_this_step", True)
                        args._log_this_step = False
                        try:
                            for i in range(0, patch_count, pb):
                                chunk_patch_ids = selected_patch_ids[i:i+pb]
                                chunk_patch_ids_list = chunk_patch_ids.detach().cpu().tolist()
                                patch_xyz = patch_info["patch_xyz"].index_select(0, chunk_patch_ids)
                                patch_attr = patch_info["patch_attr"].index_select(0, chunk_patch_ids)
                                patch_centroid = patch_info["patch_centroid"].index_select(0, chunk_patch_ids)
                                patch_scale = patch_info["patch_scale"].index_select(0, chunk_patch_ids)
                                patch_cache_keys = [ f"{cache_key}|patch={patch_id}" for patch_id in chunk_patch_ids_list]
                                ( gen_chunk, L_attr_chunk, L_policy_chunk, L_actuator_chunk, final_w_chunk, Lp_out_chunk, La_fit_chunk, La_rep_chunk, _, patch_meta_chunk) = model.forward( patch_xyz, patch_attr, cache_key=patch_cache_keys, return_patch_meta=True, coord_scale=patch_scale, return_attr_output=False)
                                if detail_log_this_step:
                                    base_model = model.module if hasattr(model, "module") else model
                                    encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))
                                gen_chunk = denormalize_patch_output( gen_chunk, patch_centroid, patch_scale)
                                chunk_size = patch_xyz.shape[0]
                                geom_groups = {}
                                L_attr = L_attr + L_attr_chunk * chunk_size
                                L_policy = L_policy + L_policy_chunk * chunk_size
                                L_actuator = L_actuator + L_actuator_chunk * chunk_size
                                Lp_out = Lp_out + Lp_out_chunk * chunk_size
                                La_fit = La_fit + La_fit_chunk * chunk_size
                                La_rep = La_rep + La_rep_chunk * chunk_size

                                for local_idx in range(chunk_size):
                                    patch_id = int(chunk_patch_ids_list[local_idx])
                                    patch_input_idx = patch_info["patch_input_idx"][patch_id]
                                    owned_input_mask = patch_info["owned_input_mask"][patch_id]
                                    anchor_idx_local = patch_meta_chunk["anchor_idx_local"][local_idx].clamp_(0, patch_input_idx.shape[0] - 1)
                                    valid_mask = patch_meta_chunk["output_valid_mask"][local_idx]
                                    owned_output_mask = owned_input_mask.index_select(0, anchor_idx_local)
                                    select_mask = valid_mask & owned_output_mask
                                    selected_pts = gen_chunk[local_idx, :, select_mask]
                                    selected_w = None
                                    if final_w_chunk is not None:
                                        selected_w = final_w_chunk[local_idx, :, select_mask]
                                    represented_owned_mask = torch.zeros_like(owned_input_mask)
                                    if select_mask.any():
                                        represented_owned_mask[anchor_idx_local[select_mask]] = True
                                    missing_owned_mask = owned_input_mask & (~represented_owned_mask)
                                    fallback_pts = None
                                    fallback_w = None
                                    if missing_owned_mask.any():
                                        patch_input_xyz_world = (patch_info["patch_centroid"][patch_id:patch_id+1] + patch_info["patch_xyz"][patch_id:patch_id+1] * patch_info["patch_scale"][patch_id:patch_id+1])
                                        fallback_pts = patch_input_xyz_world[0, :, missing_owned_mask]
                                        if final_w_chunk is not None:
                                            fallback_w = final_w_chunk.new_ones((1, int(missing_owned_mask.sum().item())))

                                    owned_local_idx = torch.nonzero(owned_input_mask, as_tuple=False).flatten()
                                    owned_global_idx = None
                                    owned_out_label = None
                                    if owned_local_idx.numel() > 0:
                                        owned_global_idx = patch_input_idx.index_select(0, owned_local_idx)
                                        if patch_meta_chunk["out_label"] is not None:
                                            owned_out_label = patch_meta_chunk["out_label"][local_idx, owned_local_idx]

                                    if valid_mask.any():
                                        gen_patch_valid = gen_chunk[local_idx:local_idx+1, :3, valid_mask]
                                        if str(getattr(args, "discretelossmode", "hard")).strip().lower() == "hard":
                                            final_w_owned = None
                                        else:
                                            final_w_owned = None if final_w_chunk is None else final_w_chunk[local_idx:local_idx+1, :, valid_mask]

                                        gt_patch_owned = input_pcd[:, :3, patch_input_idx[owned_input_mask]].contiguous()
                                        local_weight = float(max(int(owned_input_mask.sum().item()), 1))
                                        can_batch_geom = ( owned_out_label is None or int(torch.count_nonzero(owned_out_label).detach().cpu()) == 0)
                                        if can_batch_geom:
                                            geom_key = ( int(gen_patch_valid.shape[-1]), int(gt_patch_owned.shape[-1]), final_w_owned is not None)
                                            group = geom_groups.get(geom_key)
                                            if group is None:
                                                group = { "gen": [], "gt": [], "final_w": [] if final_w_owned is not None else None, "weight": 0.0}
                                                geom_groups[geom_key] = group
                                            group["gen"].append(gen_patch_valid)
                                            group["gt"].append(gt_patch_owned)
                                            if final_w_owned is not None:
                                                group["final_w"].append(final_w_owned)
                                            group["weight"] += local_weight
                                        else:
                                            out_label_owned = owned_out_label.unsqueeze(0)
                                            L_geom = L_geom + loss.get_geometry_loss(
                                                args,
                                                gen_pts=gen_patch_valid,
                                                gt_pts=gt_patch_owned,
                                                final_w=final_w_owned,
                                                out_label=out_label_owned,
                                            ) * local_weight
                                            geom_weight_sum += local_weight
                                    patch_outputs.append( { "patch_id": patch_id, "selected_pts": selected_pts, "selected_w": selected_w, "fallback_pts": fallback_pts, "fallback_w": fallback_w, "owned_global_idx": owned_global_idx, "owned_out_label": owned_out_label, "patch_meta": { "anchor_idx_local": anchor_idx_local, "output_valid_mask": valid_mask, "out_label": None if patch_meta_chunk["out_label"] is None else patch_meta_chunk["out_label"][local_idx]}})
                                geom_chunk, geom_chunk_weight = accumulate_grouped_patch_geometry( geom_groups, loss, args)
                                if geom_chunk is not None and geom_chunk_weight > 0.0:
                                    L_geom = L_geom + geom_chunk
                                    geom_weight_sum += geom_chunk_weight
                        finally:
                            args._log_this_step = prev_patch_geom_log
                        if subset_step:
                            gen_pts, compression_gt_pts, final_w, out_label = merge_patch_subset_outputs( patch_info, patch_outputs, input_pcd=input_pcd, device=input_pcd.device, dtype=input_pcd.dtype)
                            compression_cache_key = make_patch_subset_cache_key( cache_key, selected_patch_ids, total_patch_count=total_patch_count)
                        else:
                            gen_pts, final_w, out_label = merge_patch_outputs( patch_info, patch_outputs, device=input_pcd.device, dtype=input_pcd.dtype)
                            compression_gt_pts = input_xyz

                        norm = float(max(patch_count, 1))
                        L_attr = L_attr / norm
                        L_policy = L_policy / norm
                        L_actuator = L_actuator / norm
                        Lp_out = Lp_out / norm
                        La_fit = La_fit / norm
                        La_rep = La_rep / norm
                        if geom_weight_sum > 0:
                            L_geom = L_geom / geom_weight_sum
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = summarize_point_edits( input_xyz=compression_gt_pts[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)

                else:
                    optimizer.zero_grad(set_to_none=True)
                    args._log_this_step = bool(getattr(args, "verbose_step_logs", False) and detail_log_this_step)
                    encoder_debug_chunks = [] if detail_log_this_step else None
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                    with autocast_ctx:
                        gen_patches, L_attr, L_policy, L_actuator, final_w, Lp_out, La_fit, La_rep, out_label = model.forward( patches, None, cache_key=cache_key, coord_scale=fd_xyz, return_attr_output=False)
                    if detail_log_this_step:
                        base_model = model.module if hasattr(model, "module") else model
                        encoder_debug_chunks.append(dict(getattr(base_model, "last_encoder_debug", {}) or {}))

                    # 元スケールに戻す
                    gen_xyz = centroid_xyz + gen_patches[:, :3, :] * fd_xyz
                    gen_pts = gen_xyz.contiguous()
                    gen_xyz = gen_pts[:, :3, :]
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)
                    L_geom = None

                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_model_end = time.time()

                # ---------- Loss計算と最適化 ----------
                if timing_enabled:
                    timing_loss_start = time.time()
                autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp) if use_cuda else nullcontext()
                with autocast_ctx:
                    final_w_for_loss = None
                    if str(getattr(args, "discretelossmode", "hard")).strip().lower() != "hard":
                        final_w_for_loss = final_w
                    if timing_enabled:
                        sync_for_timing(use_cuda)
                        timing_noise_start = time.time()
                    if subtree_mode:
                        compression_gen_xyz = gen_xyz
                    else:
                        # 入力や診断前ではなく、編集後・量子化前にだけ一様ノイズを加える。
                        # 形状損失はcleanなgen_xyz、rate/structure損失はcompression_gen_xyzを見る。
                        compression_gen_xyz, noise_debug = prepare_compression_points( gen_xyz, args, model, collect_stats=bool(log_this_step or profile_this_step))
                    if timing_enabled:
                        sync_for_timing(use_cuda)
                        timing_noise_end = time.time()
                    if subtree_mode:
                        pass
                    elif args.split2patch:
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( args, gen_xyz=compression_gen_xyz, gt_xyz=compression_gt_pts[:, :3, :], final_w=final_w_for_loss, cache_key=compression_cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_xyz)
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}
                    else:
                        L_geom = loss.get_geometry_loss( args, gen_pts=gen_xyz, gt_pts=input_xyz, final_w=final_w_for_loss, out_label=out_label)
                        if compute_compression:
                            L_com, loss_bit, loss_single, loss_nodes, _, _ = loss.get_compression_loss( args, gen_xyz=compression_gen_xyz, gt_xyz=input_xyz[:, :3, :], final_w=final_w_for_loss, cache_key=cache_key, refresh_actual_gen=refresh_actual_gen, actual_gen_xyz=gen_xyz)
                        else:
                            zero = gen_xyz.new_zeros(())
                            L_com = zero
                            loss_bit = zero
                            loss_single = zero
                            loss_nodes = zero
                            loss.last_compression_debug = {}
                            loss.last_compression_terms = {}

                if compute_compression:
                    comp_debug_for_noise = getattr(loss, "last_compression_debug", {}) or {}
                    comp_debug_for_noise.update( { "uniform_noise_enabled": bool(noise_debug.get("enabled", False)), "uniform_noise_applied": bool(noise_debug.get("applied", False)), "uniform_noise_delta": float(noise_debug.get("delta", 0.0)), "uniform_noise_mean_abs": float(noise_debug.get("mean_abs", 0.0)), "compression_input_noisy": bool(noise_debug.get("applied", False))})
                    loss.last_compression_debug = comp_debug_for_noise

                # legacy_totalは既存互換の総合lossとして残す。
                # compression_primaryはactual codec値ではなく、last_compression_termsのgrad tensorを主目的に使う。
                terms = getattr(loss, "last_compression_terms", {}) or {}
                actual_total_bit_backend = uses_actual_total_bit_objective(args)
                if actual_total_bit_backend:
                    L_com_objective = float(getattr(args, "w_com", 1.0)) * L_com
                else:
                    bit_term = terms.get("bit", L_com.new_zeros(()))
                    single_term = terms.get("single", L_com.new_zeros(()))
                    node_term = terms.get("node", L_com.new_zeros(()))
                    bpn_term = terms.get("bpn", L_com.new_zeros(()))
                    sparsepcgc_term = terms.get("sparsepcgc", L_com.new_zeros(()))
                    lowprob_term = La_fit if torch.is_tensor(La_fit) else L_com.new_zeros(())
                    L_com_objective = float(getattr(args, "w_com", 1.0)) * ( float(getattr(args, "com_bit", 0.0)) * bit_term + float(getattr(args, "com_sin", 0.0)) * single_term + float(getattr(args, "com_node", 0.0)) * node_term + float(getattr(args, "com_bpn", 0.0)) * bpn_term + float(getattr(args, "com_sparsepcgc", 0.0)) * sparsepcgc_term + float(getattr(args, "com_lowprob", 0.0)) * lowprob_term)
                legacy_L_downstream = ( stage_factors["geom"] * args.w_geom * L_geom + stage_factors["com"] * L_com_objective)
                legacy_L_total = ( legacy_L_downstream + stage_factors["attr"] * args.w_attr * L_attr + stage_factors["policy"] * args.w_policy * L_policy + stage_factors["repair"] * args.w_actuator * L_actuator)
                L = legacy_L_total
                L_downstream = legacy_L_downstream
                L_discrete_policy = L.new_zeros(())
                cp_debug = {}
                if compression_primary_mode:
                    L, L_com_objective, cp_debug = build_compression_primary_loss( args, terms=terms, L_com=L_com, L_geom=L_geom, L_actuator=L_actuator, global_train_step=global_train_step, stage_factors=stage_factors)
                    L_downstream = L_com_objective
                    L_discrete_policy = L.new_zeros(())
                elif str(getattr(args, "discretelossmode", "hard")).strip().lower() == "hard":
                    policy_loss_fn = getattr(model, "discrete_policy_loss", None)
                    if callable(policy_loss_fn):
                        L_discrete_policy = policy_loss_fn(L_downstream.detach())
                        L = L + L_discrete_policy

                comp_debug = dict(getattr(loss, "last_compression_debug", {}) or {})
                if cp_debug:
                    comp_debug.update(cp_debug)
                    loss.last_compression_debug = comp_debug
                base_model = model.module if hasattr(model, "module") else model
                structure_debug = getattr(base_model, "last_structure_debug", {}) or {}
                if train_edit_stats is None:
                    train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)
                corr_debug = update_actual_correlation_debug(args, comp_debug, L_com, codec_actual_metric_pairs)
                if corr_debug:
                    comp_debug.update(corr_debug)
                    loss.last_compression_debug = comp_debug
                    corr_value = finite_float_or_none(corr_debug.get("corr_surrogate_actual"))
                    if ( log_this_step and bool(getattr(args, "surrogate_realign_on_low_corr", False)) and corr_value is not None and corr_value < float(getattr(args, "surrogate_realign_min_corr", 0.3))):
                        writer.write( "SurrogateRealignNotice: " f"corr_surrogate_actual={corr_value:.6f} below " f"{float(getattr(args, 'surrogate_realign_min_corr', 0.3)):.6f}; " f"realign_steps={int(getattr(args, 'surrogate_realign_steps', 0))} " "(current implementation logs the trigger; extra realign steps are not run unless added later).")
                skip_optimizer_reason = None
                if ( bool(getattr(args, "skip_optimizer_on_actual_fallback", True)) and bool(comp_debug.get("actual_codec_fallback_to_proxy", False))):
                    skip_optimizer_reason = "actual_codec_fallback_to_proxy"
                    comp_debug["optimizer_skip_reason"] = skip_optimizer_reason
                    loss.last_compression_debug = comp_debug
                compression_metric_row = build_compression_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, L_com=L_com)
                operation_metric_row = build_operation_metric_row( args, global_step=global_train_step, episode=episode, epoch=epoch, step=step, stage=current_stage, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats)
                append_csv_row( metric_csv_paths.get("compression_step"), COMPRESSION_METRIC_COLUMNS, compression_metric_row)
                accumulate_compression_episode(episode_compression_sums, compression_metric_row)
                append_csv_row( metric_csv_paths.get("operation_step"), OPERATION_METRIC_COLUMNS, operation_metric_row)
                accumulate_operation_episode(episode_operation_sums, operation_metric_row)
                maybe_record_case_debug( args, writer, case_debug_path, case_debug_counts, global_step=global_train_step, episode=episode, epoch=epoch, step=step, file_path=file_path, comp_debug=comp_debug, structure_debug=structure_debug, edit_stats=train_edit_stats, L=L, L_geom=L_geom, L_com=L_com, L_actuator=L_actuator)

                if log_this_step:
                    log_step_loss( writer, step, num_steps, L, L_geom, L_com, L_com_objective, L_attr, L_policy, L_actuator, Lp_out, La_fit, La_rep, L_discrete_policy, loss_bit, loss_single, loss_nodes)
                    if cp_debug and bool(getattr(args, "cp_log_grad_terms", True)):
                        log_compression_primary_terms(writer, step, num_steps, cp_debug)

                    log_compression_stats( writer, step, num_steps, comp_debug)

                    before_node, after_node, before_single, after_single = log_compression_train_debug( writer, step, num_steps, args, comp_debug, loss, L_com)

                    log_codec_actual_correlation( writer, step, num_steps, args, comp_debug, codec_actual_metric_pairs, before_node, after_node, before_single, after_single)

                    log_sparsepcgc_train_debug( writer, step, num_steps, args, comp_debug, sparsepcgc_proxy_actual_pairs)

                    if structure_debug:
                        log_structure_debug( writer, structure_debug, step, num_steps)

                        write_structure_decision_debug( writer, f"StructureDecision step={step + 1}/{num_steps}", structure_debug)
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_loss_end = time.time()
                
                """--- どのlossがどの勾配を作っているか確認 ---"""
                # backward_and_measure("geom", args.w_geom * L_geom, model, optimizer, writer, args)                
                # backward_and_measure("com", args.w_com  * L_com,  model, optimizer, writer, args)
                # backward_and_measure("attr", args.w_attr * L_attr, model, optimizer, writer, args)
                # backward_and_measure("policy" , args.w_policy  * L_policy,  model, optimizer, writer, args)

                step_completed = False
                total_loss_finite = bool(torch.isfinite(L.detach()).all().item()) and skip_optimizer_reason is None
                param_update_snapshots = None
                amp_info = { "enabled": bool(amp_scaler_enabled), "found_inf": None, "scale_before": None, "scale_after": None, "consecutive_amp_skips": int(consecutive_amp_skips)}
                if total_loss_finite:
                    param_update_snapshots = capture_param_update_snapshots( args, model, step + 1, num_steps)
                if skip_optimizer_reason is not None:
                    writer.write( "Skipped optimizer step because actual codec teacher fell back to proxy at " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}; " "this prevents proxy-only updates from replacing real-compression imitation.")
                elif not total_loss_finite:
                    writer.write( f"Skipped optimizer step due to non-finite total loss at " f"episode={episode + 1}, epoch={epoch + 1}, step={step + 1}/{num_steps}.")
                elif amp_scaler_enabled:
                    scale_before = float(scaler.get_scale())
                    amp_info["scale_before"] = scale_before
                    scaler.scale(L).backward()
                    scaler.unscale_(optimizer)
                    grad_clip = float(getattr(args, "train_grad_clip", 0.0))
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
                    if bool(getattr(args, "debug_grad_flow", False)):
                        log_grad_flow(args, writer, model, step + 1, num_steps)
                    scaler.step(optimizer)
                    optimizer_state = scaler._per_optimizer_states[id(optimizer)]
                    found_inf = 0.0
                    if optimizer_state["found_inf_per_device"]:
                        found_inf = float( sum(v.item() for v in optimizer_state["found_inf_per_device"].values()))
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    amp_info["found_inf"] = found_inf
                    amp_info["scale_after"] = scale_after
                    step_completed = found_inf == 0.0 and scale_after >= scale_before
                    if step_completed:
                        consecutive_amp_skips = 0
                    else:
                        consecutive_amp_skips += 1
                        if consecutive_amp_skips >= amp_overflow_patience:
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
                    L.backward()
                    grad_clip = float(getattr(args, "train_grad_clip", 0.0))
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
                    log_grad_flow(args, writer, model, step + 1, num_steps)
                    optimizer.step()
                    step_completed = True
                    consecutive_amp_skips = 0
                if step_completed:
                    log_param_updates( args, writer, model, param_update_snapshots, step + 1, num_steps)
                if timing_enabled:
                    sync_for_timing(use_cuda)
                    timing_step_end = time.time()
                epoch_has_optimizer_step = epoch_has_optimizer_step or step_completed
                
                if epoch_metric_sums is None:
                    epoch_metric_sums = new_metric_sums(L.device, plot.num_loss)
                add_metric_sums( epoch_metric_sums, [ L, L_geom, L_com, L_attr, L_policy, loss_single, loss_nodes, Lp_out, La_fit, La_rep, L_actuator, *surrogate_plot_metrics(loss)], L.device)
                if episode_metric_sums is None:
                    episode_metric_sums = new_metric_sums(L.device, plot.num_loss)
                step_metric_values = [ L, L_geom, L_com, L_attr, L_policy, loss_single, loss_nodes, Lp_out, La_fit, La_rep, L_actuator, *surrogate_plot_metrics(loss)]
                add_metric_sums(episode_metric_sums, step_metric_values, L.device)
                accumulate_checkpoint_metrics( episode_checkpoint_sums, compression_metric_row, operation_metric_row, step_metric_values)
                if train_edit_stats is None:
                        train_edit_stats = summarize_point_edits( input_xyz=input_xyz[:, :3, :], gen_pts=gen_pts, final_w=final_w, args=args)
                plot.record_point_edits("step", global_train_step + 1, train_edit_stats)
                plot_step_info = plot.record_metrics("step", global_train_step + 1, step_metric_values)
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

            # lr scheduler
            lr_before_scheduler = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
            if epoch_has_optimizer_step:
                scheduler_steplr.step()
            else:
                writer.write("No successful optimizer step in this epoch; lr_scheduler.step() was skipped.")
            lr_after_scheduler = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
            if lr_after_scheduler != lr_before_scheduler:
                log_for_better_event( for_better_path, "scheduler_lr_step", global_epoch=global_epoch + 1, lr_before=lr_before_scheduler, lr_after=lr_after_scheduler)

            # ログの記録
            if epoch_metric_sums is not None:
                epoch_avgs = metric_avgs_to_floats(epoch_metric_sums)
                plot.epo_avg = epoch_avgs
                plot_epoch_info = plot.record_metrics("epo", global_epoch + 1, epoch_avgs)
                log_plot_skip_epoch( writer, plot_epoch_info, global_epoch)
                writer.write(format_metric_summary("EpochAvg", plot.metric_keys, epoch_avgs))
            epoch_edit_info = plot.record_point_edits("epo", global_epoch + 1)
            log_epoch_point_edit_average( writer, epoch_edit_info, global_epoch)
            global_epoch += 1
            plot.plot_loss_curve("step")
            plot.plot_loss_curve("epo")
            plot.plot_point_edit_curve("step")
            plot.plot_point_edit_curve("epo")
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
        log_episode_point_edit_average( writer, episode_edit_info, episode)
        plot.plot_loss_curve("epi")
        plot.plot_point_edit_curve("epi")
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