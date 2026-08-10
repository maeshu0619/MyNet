"""MyNetの訓練エントリポイント。"""

from models.utils.training.train_runtime import *

def train(model, args, loss, writer, plot, notifier=None):
    """==========================================================="""
    """セットアップ"""
    """==========================================================="""
    """基本情報"""
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False)) # ランスシードを固定し、学習結果の再現性を確保する
    best_loss = float('inf') # 後続の計算・ログのため
    if bool(getattr(args, "train_all_datasets", False)):
        raw_seq_dirs = []
        limited_by_dataset = []
        for dataset_name in getattr(args, "train_all_dataset_names", ("8i", "MVUB", "UVG")):
            dataset_seq_dirs = collect_seq_dirs2(
                args.input_dir, dataset_name=dataset_name
            )
            raw_seq_dirs.extend(dataset_seq_dirs)
            if not str(
                getattr(args, "network_k_diagnostic_sequence_name", "") or ""
            ).strip():
                limited_by_dataset.extend(
                    _limit_training_seq_dirs(
                        dataset_seq_dirs, args, dataset_name=dataset_name
                    )
                )
        if str(getattr(args, "network_k_diagnostic_sequence_name", "") or "").strip():
            seq_dirs = _limit_training_seq_dirs(raw_seq_dirs, args)
        else:
            seq_dirs = limited_by_dataset
        writer.write(
            "AllDatasetTraining: enabled=True, order={}, sequences={}".format(
                ">".join(getattr(args, "train_all_dataset_names", ())),
                len(seq_dirs),
            )
        )
    else:
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
            "Training sequence limit applied: "
            f"using {len(seq_dirs)} of {len(raw_seq_dirs)} sequence directories"
        )
        writer.write(f"Kept sequence dirs: {kept_names}")
    seq_datasets = [(seq_dir, PlyDirDataset(args, seq_dir)) for seq_dir in seq_dirs] # 各シーケンス内のPLY点群ファイルを読み込むデータセットを作る
    single_plan_teacher_store = None
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "single_plan_student":
        writer.write(
            "TrainingPerformanceContract: applied_plan=single_plan_student, "
            "actual_source=network_executable_plan, den6_apply=0, pool_apply=0, "
            "train_test_policy_match=1"
        )
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
            if str(getattr(
                args, "single_plan_training_stage", "representation"
            )).strip().lower() in {"representation", "fast_distillation"}:
                def _single_plan_setting_id(scale_sr):
                    return (
                        "native_vs{}_pq{}_ae{}_sr{}_m{}".format(
                        float(getattr(args, "sparsepcgc_voxel_size", 1.0)),
                        int(getattr(args, "sparsepcgc_pos_quantscale", 1)),
                        int(getattr(args, "sparsepcgc_scale_ae", 0)),
                        int(scale_sr),
                        int(getattr(args, "sparsepcgc_scale_m", 8)),
                        ).replace("vs1.0", "vs1")
                    )
                setting_ids = {
                    _single_plan_setting_id(
                        int(getattr(args, "sparsepcgc_scale_sr", 2))
                    )
                }
                cached_paths = set()
                matching_state_ids = set()
                for rows in single_plan_teacher_store.states.values():
                    if not rows:
                        continue
                    state = dict(rows[0].get("state_key") or {})
                    if str(state.get("setting_id", "")) in setting_ids:
                        cached_paths.add(os.path.realpath(str(state.get("input_file", ""))))
                        matching_state_ids.add(
                            "{}|{}".format(
                                str(state.get("input_sha256", "")),
                                str(state.get("setting_id", "")),
                            )
                        )
                incomplete_teacher_states = (
                    single_plan_teacher_store.incomplete_state_reasons()
                )
                matching_incomplete = {
                    state_id: reason
                    for state_id, reason in incomplete_teacher_states.items()
                    if state_id in matching_state_ids
                }
                if (
                    bool(getattr(
                        args, "single_plan_require_complete_teacher_pool", True
                    ))
                    and matching_incomplete
                ):
                    raise RuntimeError(
                        "Single-Plan Gate B未達: 現codec設定の完全candidate Pool"
                        "またはscore/rankが欠けているため、Pool外Voxelを"
                        "negative化する蒸留を停止した。欠損state例={}".format(
                            list(matching_incomplete.items())[:3]
                        )
                    )
                if matching_incomplete:
                    writer.write(
                        "SinglePlanTeacherWarning: Gate B incomplete; "
                        "using selected-positive/available-rank supervision only, "
                        "unlisted_voxel_negative=0, states={}".format(
                            len(matching_incomplete)
                        )
                    )
                selected_datasets = []
                selected_file_count = 0
                for seq_dir, dataset in seq_datasets:
                    selected_files = [
                        path for path in getattr(dataset, "files", ())
                        if os.path.realpath(path) in cached_paths
                    ]
                    if selected_files:
                        dataset.files = selected_files
                        dataset.all_files = list(selected_files)
                        selected_datasets.append((seq_dir, dataset))
                        selected_file_count += len(selected_files)
                if selected_file_count <= 0:
                    raise RuntimeError(
                        "Single-Plan offline蒸留対象と現在のdataset/codec設定が一致しない。"
                        "未知frameをnegativeやActual探索へ黙って置換しない"
                    )
                seq_datasets = selected_datasets
                writer.write(
                    "SinglePlanOfflineCoverage: stage={}, setting={}, frames={}, "
                    "uncached_frame_apply=0".format(
                        str(getattr(args, "single_plan_training_stage", "")),
                        ",".join(sorted(setting_ids)),
                        selected_file_count,
                    )
                )
        else:
            raise RuntimeError(
                "Single-Plan訓練用Layer A Cacheがない。"
                "tools/build_exact_teacher_cache.pyでfingerprint検証済みcacheを構築すること。"
                "旧datasetへのsilent fallbackは許可しない"
            )
    elif str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() == "ana_den6_online":
        writer.write(
            "TrainingPerformanceContract: applied_plan=den6_exact_rank_plus_network_residual, "
            "actual_source=heuristic_teacher_plan, network_only_performance=0, "
            "deployment_checkpoint_eligible=0"
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
    _configure_streaming_exact_caches(args, seq_datasets, writer)
    total_train_files = sum(len(dataset) for _, dataset in seq_datasets) # 全シーケンスに含まれる点群ファイル数を合計し、総Step数の見積もりなどに使用
    den6_prefetch_lookahead = max(
        int(getattr(args, "heuristic_guidance_online_prefetch_lookahead", 0)), 0
    )
    if str(getattr(args, "heuristic_guidance_mode", "")).strip().lower() not in {
        "ana_den6_online", "ana_den6_residual"
    }:
        den6_prefetch_lookahead = 0
    if bool(getattr(args, "train_all_datasets", False)):
        # 非同期workerへ可変argsを渡すと、dataset切替と競合して別dataset名の
        # cacheを作り得る。全dataset modeでは各Stepの同期cache lookupを使う。
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
    case_debug_path = (
        init_case_debug_csv(args, plot, writer)
        if bool(getattr(args, "save_step_metric_csv", False))
        else None
    ) # 詳細Step CSVを明示した場合だけGood/Bad caseを保存する
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
    if bool(getattr(args, "save_step_metric_csv", False)) and bool(getattr(args, "phase7_eval_summary", True)):
        metric_csv_paths["phase7_eval_summary"] = _phase7_eval_summary_path(args, plot)
        init_csv_file(
            metric_csv_paths["phase7_eval_summary"],
            PHASE7_EVAL_SUMMARY_COLUMNS,
            writer,
            "Phase7EvalSummaryCSV",
        )
    # 各損失項が各モジュール・点操作へ流す勾配量を記録するCSV
    step_grad_dir = getattr(plot, "save_dir", None) or getattr(args, "out_path", ".")
    metric_csv_paths["step_grad"] = None
    if bool(getattr(args, "save_step_metric_csv", False)) and bool(getattr(args, "step_grad_log", True)):
        metric_csv_paths["step_grad"] = os.path.join(step_grad_dir, f"{args.time}_MyNetwork_step_grad.csv")
        init_csv_file(metric_csv_paths["step_grad"], STEP_GRAD_COLUMNS, writer, "StepGradCSV")
        writer.write(
            "StepGradCSVMode: "
            f"first_step_only={bool(getattr(args, 'step_grad_first_step_only', True))}, "
            f"interval={int(getattr(args, 'step_grad_log_interval', 1))}"
        )
    else:
        writer.write("StepGradCSV: disabled (Episode/100-Step plot mode)")

    """原因診断のためのログ"""
    memory_diagnostics_path = os.path.join(
        step_grad_dir, f"{args.time}_memory_diagnostics.csv"
    )
    memory_diagnostics = MemoryDiagnosticsCSV(memory_diagnostics_path)

    _record_memory = partial(
        record_training_memory,
        memory_diagnostics,
        args=args,
        model=model,
        loss=loss,
        writer=writer,
    )

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
    emulator_optimizer, emulator_scheduler = build_emulator_optimizer_and_scheduler(
        model, args, writer
    )
    apply_optimizer_lr_floor(optimizer, args, label="main", writer=writer, global_step=0, reason="train_start") # main LRが開始時点からfloor未満なら下限値へ戻す
    amp_state = setup_amp( model, args, writer) # CUDA利用可否
    use_cuda = amp_state["use_cuda"] # GPU使用の有無
    use_amp = amp_state["use_amp"] # 自動混合精度で計算するか否か
    amp_dtype = amp_state["amp_dtype"] # AMPで使う浮動小数点型の保存
    amp_scaler_enabled = amp_state["amp_scaler_enabled"] # GradScalerを使うのか否か
    scaler = amp_state["scaler"] # AMPのGradScaler。AMPでスケーリングされた勾配を逆スケーリングしてOptimizerに渡すために使う
    emulator_scaler = torch.cuda.amp.GradScaler(
        enabled=bool(amp_scaler_enabled and emulator_optimizer is not None),
        init_scale=float(getattr(args, "amp_init_scale", 1.0)),
    )
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
    if emulator_optimizer is not None:
        emulator_optimizer.zero_grad(set_to_none=True)

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
    # 生の補助損失が下がった際にbalance係数を逆増幅すると、全体損失へ
    # 改善が現れない。訓練中は一度締めた係数を再び大きくしない。
    tail_support_balance_scale_state = float("nan")
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
            epoch_dataset_name = getattr(
                dataset,
                "training_dataset_name",
                training_dataset_name_from_path(seq_dir),
            )
            dataset_context_changed = activate_training_dataset_context(
                args, epoch_dataset_name
            )
            writer.write(f"⦿⦿⦿ Epoch {epoch + 1}/{num_seq} : {seq_dir} ⦿⦿⦿")
            if dataset_context_changed:
                writer.write(
                    "TrainingDatasetContext: dataset={}, shared_codec=1, "
                    "native_depth={}, AE={}, SR={}, m={}, resolution={}".format(
                        args.dataname,
                        int(getattr(args, "sparsepcgc_native_bit_depth", 0)),
                        int(getattr(args, "sparsepcgc_scale_ae", 0)),
                        int(getattr(args, "sparsepcgc_scale_sr", 0)),
                        int(getattr(args, "sparsepcgc_scale_m", 0)),
                        int(getattr(args, "sparsepcgc_psnr_resolution", 0)),
                    )
                )
            sequence_basename = os.path.basename(os.path.normpath(str(seq_dir)))
            sequence_name = (
                f"{args.dataname}/{sequence_basename}"
                if bool(getattr(args, "train_all_datasets", False))
                else sequence_basename
            )
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
                if emulator_optimizer is not None:
                    emulator_optimizer.zero_grad(set_to_none=True)
                emulator_loss = None
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
                if (
                    heuristic_mode == "single_plan_student"
                    and compression_debug_terms.get(
                        "actual_total_bit_percent_fresh", None
                    ) is not None
                ):
                    # compression backendでいうteacherは「Actual codec scalar」を
                    # 指すため、行動Teacher planと誤読されないsource名へ上書きする。
                    compression_debug_terms["actual_value_source"] = "fresh_network_plan"
                    compression_debug_terms["policy_action_source"] = "single_plan_student"
                    loss.last_compression_debug = dict(compression_debug_terms)
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
                    proposed_tail_support_scale = float(tail_balance["scale"])
                    tail_primary_mag = tail_balance.get("primary_mag", None)
                    tail_primary_is_valid = (
                        tail_primary_mag is not None
                        and math.isfinite(float(tail_primary_mag))
                        and float(tail_primary_mag) > 1e-8
                    )
                    if tail_primary_is_valid:
                        tail_support_balance_scale_state = monotonic_support_scale(
                            tail_support_balance_scale_state,
                            proposed_tail_support_scale,
                        )
                    tail_support_scale = (
                        float(tail_support_balance_scale_state)
                        if math.isfinite(float(tail_support_balance_scale_state))
                        else proposed_tail_support_scale
                    )
                    total_support_balance = _compression_primary_remaining_support_balance(
                        args,
                        compression_support_anchor if torch.is_tensor(compression_support_anchor) else L,
                        abs(float(cp_debug.get("cp_aux_block_scaled", 0.0)))
                        if isinstance(cp_debug, dict) else 0.0,
                        tail_support_scale * tail_support_raw,
                        enabled=uses_actual_total_bit_objective(args),
                    )
                    tail_support_scale *= float(total_support_balance["scale"])
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
                        cp_debug["cp_support_tail_proposed_scale"] = float(
                            proposed_tail_support_scale
                        )
                        cp_debug["cp_support_tail_reason"] = str(tail_balance.get("reason", ""))
                        cp_debug["cp_support_total_balance_reason"] = str(
                            total_support_balance.get("reason", "")
                        )
                        cp_debug["cp_support_total_target_ratio"] = float(
                            total_support_balance.get("target_ratio", float("nan"))
                        )
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
                    # このmodeではActuatorへ適用されたplanもActual入力もStudent由来である。
                    # Shadow TeacherのActualを数えず、推論と同じ経路の校正履歴だけを
                    # checkpointへ保存する。
                    current_loss_debug = dict(
                        getattr(loss, "last_compression_debug", {}) or {}
                    )
                    student_actual_percent = finite_float_or_none(
                        compression_debug_terms.get(
                            "actual_total_bit_percent_fresh",
                            current_loss_debug.get(
                                "actual_total_bit_percent_fresh",
                                current_loss_debug.get(
                                    "actual_total_bit_percent", None
                                ),
                            ),
                        )
                    )
                    if student_actual_percent is not None:
                        base_student = model.module if hasattr(model, "module") else model
                        with torch.no_grad():
                            base_student.single_plan_actual_training_updates.add_(1)
                        compression_debug_terms.update({
                            "single_plan_actual_training_update": True,
                            "single_plan_actual_training_updates": int(
                                base_student.single_plan_actual_training_updates.detach().cpu()
                            ),
                            "single_plan_actual_compression_percent": float(
                                student_actual_percent
                            ),
                            "single_plan_actual_source": "network_executable_plan",
                            # ``fresh_teacher`` はSurrogate内部で「fresh Actual教師」
                            # を表す旧名称だが、Heuristic Teacherとの混同を避け、
                            # 表示上はStudent実行planのActualであることを明示する。
                            "actual_value_source": "fresh_student_actual",
                        })
                        current_loss_debug.update(compression_debug_terms)
                        loss.last_compression_debug = current_loss_debug
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
                    # Exact主Lossへ混ぜず、独立optimizerでEmulatorだけを更新する。
                    emulator_loss = shadow_distill
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
                emulator_step_completed = False
                if emulator_optimizer is not None and torch.is_tensor(emulator_loss):
                    emulator_parameters = [
                        parameter
                        for group in emulator_optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.requires_grad
                    ]
                    emulator_before = [
                        parameter.detach().clone()
                        for parameter in emulator_parameters
                    ]
                    emulator_finite = bool(
                        torch.isfinite(emulator_loss.detach()).all().item()
                    )
                    if emulator_finite:
                        if bool(emulator_scaler.is_enabled()):
                            emulator_scaler.scale(emulator_loss).backward()
                            emulator_scaler.unscale_(emulator_optimizer)
                            emulator_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                                emulator_parameters,
                                max_norm=float(getattr(
                                    args, "single_plan_student_grad_clip", 1.0
                                )),
                            ))
                            emulator_scaler.step(emulator_optimizer)
                            emulator_scaler.update()
                        else:
                            emulator_loss.backward()
                            emulator_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                                emulator_parameters,
                                max_norm=float(getattr(
                                    args, "single_plan_student_grad_clip", 1.0
                                )),
                            ))
                            emulator_optimizer.step()
                        emulator_step_completed = True
                        emulator_update_norm = math.sqrt(sum(
                            float(
                                (parameter.detach() - before).float().pow(2).sum().cpu()
                            )
                            for parameter, before in zip(
                                emulator_parameters, emulator_before
                            )
                        ))
                    else:
                        emulator_grad_norm = float("nan")
                        emulator_update_norm = 0.0
                    compression_debug_terms.update({
                        "fast_emulator_optimizer_separate": True,
                        "fast_emulator_optimizer_step": bool(emulator_step_completed),
                        "fast_emulator_loss": float(
                            emulator_loss.detach().float().cpu()
                        ),
                        "fast_emulator_grad_norm_raw": float(emulator_grad_norm),
                        "fast_emulator_parameter_update_norm": float(
                            emulator_update_norm
                        ),
                    })
                    comp_debug.update({
                        key: compression_debug_terms[key]
                        for key in (
                            "fast_emulator_optimizer_separate",
                            "fast_emulator_optimizer_step",
                            "fast_emulator_loss",
                            "fast_emulator_grad_norm_raw",
                            "fast_emulator_parameter_update_norm",
                        )
                    })
                    loss.last_compression_debug = comp_debug
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
                if bool(getattr(args, "save_step_metric_csv", False)) and _phase7_should_save_eval_summary(args, global_train_step):
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
                plot.record_metrics("step", global_train_step + 1, step_metric_values) # Episode/100 Step平均だけをメモリ上で集計
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
                        f"guidance_disk_hit={bool(audit_plan.get('guidance_disk_tensor_cache_hit', False))}, "
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
                        f", emulator=(separate={bool(audit_compression.get('fast_emulator_optimizer_separate', False))}, "
                        f"step={bool(audit_compression.get('fast_emulator_optimizer_step', False))}, "
                        f"grad={float(audit_compression.get('fast_emulator_grad_norm_raw', 0.0) or 0.0):.6g}, "
                        f"update={float(audit_compression.get('fast_emulator_parameter_update_norm', 0.0) or 0.0):.6g}, "
                        f"prune_topm={float(audit_compression.get('single_plan_shadow_prune_source_topm_coverage', 0.0) or 0.0):.4f}, "
                        f"adjust_topm={float(audit_compression.get('single_plan_shadow_adjust_source_topm_coverage', 0.0) or 0.0):.4f}, "
                        f"direction={float(audit_compression.get('single_plan_shadow_adjust_direction_recall', 0.0) or 0.0):.4f}, "
                        f"prune_feature_oracle={float(audit_compression.get('single_plan_shadow_prune_fixed_feature_oracle_recall', 0.0) or 0.0):.4f}, "
                        f"adjust_feature_oracle={float(audit_compression.get('single_plan_shadow_adjust_fixed_feature_oracle_recall', 0.0) or 0.0):.4f}, "
                        f"prune_rank={float(audit_compression.get('single_plan_shadow_prune_rank_spearman', 0.0) or 0.0):.4f}, "
                        f"adjust_rank={float(audit_compression.get('single_plan_shadow_adjust_rank_spearman', 0.0) or 0.0):.4f})"
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
                        "single_plan_actual_training_updates": int(
                            getattr(
                                audit_model,
                                "single_plan_actual_training_updates",
                                torch.zeros((), dtype=torch.long),
                            ).detach().cpu()
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
                    diversity["ratios"].append(audit_state_scalar(audit_state, "network_only_total_ratio_unconstrained"))
                    diversity["shares"].append(audit_state_list(audit_state, "network_only_shares_raw"))
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
                        f"shares_raw={audit_state_list(audit_state, 'network_only_shares_raw')}, "
                        f"shares_hard={audit_state_list(audit_state, 'network_only_shares')}, "
                        f"shares_mean={audit_state_list(audit_state, 'network_only_shares_mean')}, "
                        f"total_ratio_raw={audit_state_scalar(audit_state, 'network_only_total_ratio_raw'):.8f}, "
                        f"total_ratio_unconstrained={audit_state_scalar(audit_state, 'network_only_total_ratio_unconstrained'):.8f}, "
                        f"total_ratio_hard={audit_state_scalar(audit_state, 'network_only_total_ratio'):.8f}, "
                        f"total_ratio_mean={audit_state_scalar(audit_state, 'network_only_total_ratio_mean'):.8f}, "
                        f"gates={audit_state_list(audit_state, 'operation_gate_hard')}, "
                        f"priorities={audit_state_list(audit_state, 'network_only_priorities')}, "
                        f"temperature={audit_state_scalar(audit_state, 'network_only_temperature'):.6f}, "
                        f"threshold={audit_state_list(audit_state, 'network_only_where_threshold')}, "
                        f"exploration_fraction={audit_state_scalar(audit_state, 'network_only_exploration_fraction'):.6f}, "
                        f"diversity=(plan_hash={plan_hash}, unique_rate={unique_plan_rate:.6f}, "
                        f"repeat_rate={same_plan_repeat_rate:.6f}, jaccard_distance={where_jaccard_distance:.6f}, "
                        f"ratio_std={ratio_std:.8f}, share_std={share_std}, "
                        f"priority_order={list(audit_plan.get('priority_order', []))}, "
                        f"add_direction_hist={add_direction_hist}, adjust_direction_hist={adjust_direction_hist}), "
                        f"entropy=(where={audit_state_scalar(audit_state, 'network_only_where_entropy'):.6g}, "
                        f"amount={audit_state_scalar(audit_state, 'network_only_amount_entropy'):.6g}, "
                        f"action={audit_state_scalar(audit_state, 'network_only_action_entropy'):.6g}, "
                        f"direction={audit_state_scalar(audit_state, 'network_only_direction_entropy'):.6g}), "
                        f"gain=(local={audit_state_scalar(audit_state, 'network_only_predicted_local_gain_sum'):.6g}, "
                        f"interaction={audit_state_scalar(audit_state, 'network_only_interaction_correction'):.6g}, "
                        f"plan={audit_state_scalar(audit_state, 'network_only_predicted_plan_gain'):.6g}, "
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
                            f"prune_topm={float(single_distill_debug.get('prune_source_topm_coverage', 0.0) or 0.0):.6g}, "
                            f"adjust_topm={float(single_distill_debug.get('adjust_source_topm_coverage', 0.0) or 0.0):.6g}, "
                            f"prune_rank_r={float(single_distill_debug.get('prune_rank_spearman', float('nan'))):.6g}, "
                            f"adjust_rank_r={float(single_distill_debug.get('adjust_rank_spearman', float('nan'))):.6g}, "
                            f"score_loss=({float(single_distill_debug.get('prune_score_loss', 0.0) or 0.0):.6g},"
                            f"{float(single_distill_debug.get('add_score_loss', 0.0) or 0.0):.6g},"
                            f"{float(single_distill_debug.get('adjust_score_loss', 0.0) or 0.0):.6g}), "
                            f"rank_loss=({float(single_distill_debug.get('prune_rank_loss', 0.0) or 0.0):.6g},"
                            f"{float(single_distill_debug.get('add_rank_loss', 0.0) or 0.0):.6g},"
                            f"{float(single_distill_debug.get('adjust_rank_loss', 0.0) or 0.0):.6g}), "
                            f"prune_fixed_oracle={float(single_distill_debug.get('prune_fixed_feature_oracle_recall', 0.0) or 0.0):.6g}, "
                            f"adjust_fixed_oracle={float(single_distill_debug.get('adjust_fixed_feature_oracle_recall', 0.0) or 0.0):.6g}, "
                            f"prune_recall={float(single_distill_debug.get('prune_source_recall', 0.0) or 0.0):.6g}, "
                            f"add_target_recall={float(single_distill_debug.get('add_target_recall', 0.0) or 0.0):.6g}, "
                            f"adjust_recall={float(single_distill_debug.get('adjust_source_recall', 0.0) or 0.0):.6g}, "
                            f"direction_recall={float(single_distill_debug.get('adjust_direction_recall', 0.0) or 0.0):.6g}, "
                            f"teacher_role={single_distill_debug.get('teacher_role', '')}"
                        )
                    if heuristic_mode == "network_k_proposal_policy":
                        selected_slot_value = int(round(audit_state_scalar(audit_state, "k_proposal_selected_slot")))
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
                            f"expected_counts={audit_state_list(audit_state, 'k_proposal_expected_count')}, "
                            f"executed_counts={audit_state_list(audit_state, 'k_proposal_executed_count')}, "
                            f"execution_count_mismatch={audit_state_list(audit_state, 'k_proposal_execution_count_mismatch')}, "
                            f"critic_gain={audit_state_list(audit_state, 'k_proposal_predicted_gain')}, "
                            f"critic_geometry={audit_state_list(audit_state, 'k_proposal_predicted_geometry')}, "
                            f"critic_interaction={audit_state_list(audit_state, 'k_proposal_predicted_interaction')}, "
                            f"critic_uncertainty={audit_state_list(audit_state, 'k_proposal_uncertainty')}"
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
                if one_plan_full_cloud:
                    # den6だけでなくNetwork-only/Single-Planも同じfull-cloud
                    # autograd graphとsaved-tensor offloadを作る。旧条件では
                    # Single-Plan時だけ前StepのCPU payloadが残り続けていた。
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
                    released_autograd_refs = release_autograd_transient_references(
                        model=model,
                        loss=loss,
                        args=args,
                    )
                    if released_autograd_refs and global_train_step == 0:
                        writer.write(
                            "AutogradTransientRelease: "
                            + ", ".join(released_autograd_refs[:24])
                        )
                    # backward・勾配監査・ログが全て完了したため、autograd nodeが
                    # Python側に残っていてもCPUへ退避した巨大payloadは不要。
                    offload_release = release_saved_tensor_offload_payloads()
                    if (
                        global_train_step == 0
                        and int(offload_release.get("released_count", 0)) > 0
                    ):
                        writer.write(
                            "SavedTensorOffloadRelease: "
                            f"count={int(offload_release['released_count'])}, "
                            f"mb={float(offload_release['released_bytes']) / (1024.0 ** 2):.3f}"
                        )
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _release_cpu_step_memory()
                    # 解放漏れが再発してもOS全体のRAMを食い切る前に停止する。
                    # 直近実走では約70MB/Stepが残り、最終的にSSHまで不通になった。
                    offload_after_cleanup = saved_tensor_offload_stats()
                    outstanding_offload_bytes = int(
                        offload_after_cleanup.get("outstanding_bytes", 0)
                    )
                    previous_offload_bytes = int(getattr(
                        args, "_saved_tensor_offload_previous_cleanup_bytes", 0
                    ))
                    leak_steps = int(getattr(
                        args, "_saved_tensor_offload_growth_steps", 0
                    ))
                    if outstanding_offload_bytes > previous_offload_bytes + 16 * 1024 * 1024:
                        leak_steps += 1
                    else:
                        leak_steps = 0
                    setattr(
                        args,
                        "_saved_tensor_offload_previous_cleanup_bytes",
                        outstanding_offload_bytes,
                    )
                    setattr(args, "_saved_tensor_offload_growth_steps", leak_steps)
                    if (
                        outstanding_offload_bytes >= 1024 * 1024 * 1024
                        and leak_steps >= 3
                    ):
                        raise RuntimeError(
                            "saved-tensor CPU offload leak guard: "
                            f"{outstanding_offload_bytes / (1024.0 ** 2):.1f}MB "
                            f"remains after cleanup and grew for {leak_steps} steps. "
                            "Training was stopped before the Linux OOM killer could "
                            "terminate Python/SSH. Check *_memory_diagnostics.csv."
                        )
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
                if emulator_scheduler is not None and emulator_optimizer is not None:
                    # main schedulerと同じ有効フラグを通す。従来はここだけ毎Epoch
                    # 無条件にstepしており、長期訓練でEmulator LRがほぼ0になっていた。
                    emulator_scheduler_event = step_scheduler_with_floor(
                        emulator_scheduler,
                        emulator_optimizer,
                        args,
                        writer=writer,
                        global_epoch=global_epoch + 1,
                        global_step=global_train_step,
                    )
                    scheduler_event["emulator_scheduler_stepped"] = bool(
                        emulator_scheduler_event.get("scheduler_stepped", False)
                    )
                    scheduler_event["current_lr_emulator"] = optimizer_lrs_safe(
                        emulator_optimizer
                    )
                if scheduler_event.get("scheduler_stepped"):
                    scheduler_step_count += 1
                scheduler_event["scheduler_step_count"] = scheduler_step_count
                scheduler_event["current_lr_main"] = optimizer_lrs_safe(optimizer)
                scheduler_event["current_lr_surrogate"] = optimizer_lrs_safe(getattr(loss, "surrogate_optimizer", None))
                log_for_better_event( for_better_path, "scheduler_lr_step", **scheduler_event)
            else:
                writer.write("No successful optimizer step in this epoch; lr_scheduler.step() was skipped.")

            global_epoch += 1
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
        _record_memory(
            "episode_before_plots",
            episode=episode + 1,
            global_step=global_train_step,
        )
        plot.plot_loss_curve("epi")
        plot.plot_point_edit_curve("epi")
        plot.plot_loss_curve("step100")
        plot.plot_point_edit_curve("step100")
        _record_memory(
            "episode_after_plots",
            episode=episode + 1,
            global_step=global_train_step,
        )
        writer.write(f"Saved episode CSV/plots and 100-step average plots: {plot.save_dir}")
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
        checkpoint_metrics["full_cloud_val_geometry"] = full_cloud_val.get("geometry_value")
        checkpoint_metrics["full_cloud_val_fixed_objective"] = full_cloud_val.get("fixed_objective")
        checkpoint_metrics["full_cloud_val_actual_count"] = int(full_cloud_val.get("count") or 0)
        checkpoint_metrics["full_cloud_val_sample_signature"] = str(
            full_cloud_val.get("sample_signature") or ""
        )
        fixed_validation_geometry = finite_float_or_none(
            full_cloud_val.get("geometry_value")
        )
        if fixed_validation_geometry is not None:
            fixed_geom_reference = checkpoint_gate_refs.get("fixed_validation_geom")
            if fixed_geom_reference is None:
                fixed_geom_reference = float(fixed_validation_geometry)
                checkpoint_gate_refs["fixed_validation_geom"] = fixed_geom_reference
            relative_limit = float(getattr(args, "checkpoint_geom_rel_factor", 1.5))
            absolute_limit = float(getattr(args, "checkpoint_geom_abs_max", 0.0))
            fixed_geometry_ok = bool(
                (relative_limit <= 0.0 or fixed_validation_geometry <= abs(fixed_geom_reference) * relative_limit)
                and (absolute_limit <= 0.0 or fixed_validation_geometry <= absolute_limit)
            )
            checkpoint_metrics["fixed_validation_geom_reference"] = float(
                fixed_geom_reference
            )
            checkpoint_metrics["geometry_ok"] = fixed_geometry_ok
            checkpoint_metrics["safety_ok"] = bool(
                fixed_geometry_ok
                and checkpoint_metrics.get("repair_ok", True)
                and checkpoint_metrics.get("node_ok", True)
                and checkpoint_metrics.get("single_ok", True)
                and checkpoint_metrics.get("operation_ok", True)
            )
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
        single_plan_actual_gate_ok = True
        if (
            str(getattr(args, "heuristic_guidance_mode", "")).strip().lower()
            == "single_plan_student"
            and str(getattr(
                args, "single_plan_training_stage", "representation"
            )).strip().lower() == "actual_calibration"
        ):
            base_student = model.module if hasattr(model, "module") else model
            actual_eval_count = int(
                base_student.single_plan_actual_training_updates.detach().cpu()
            )
            checkpoint_delta = finite_float_or_none(
                checkpoint_metrics.get("checkpoint_actual_delta")
            )
            validation_delta = finite_float_or_none(
                checkpoint_metrics.get("full_cloud_val_actual_percent")
            )
            operation_nonzero = all(
                float(checkpoint_metrics.get(name, 0.0) or 0.0) > 0.0
                for name in (
                    "added_ratio_percent",
                    "deleted_ratio_percent",
                    "adjusted_ratio_percent",
                )
            )
            single_plan_actual_gate_ok = bool(
                actual_eval_count > 0
                and checkpoint_delta is not None
                and checkpoint_delta < 0.0
                and validation_delta is not None
                and validation_delta < 0.0
                and operation_nonzero
            )
            if actual_eval_count <= 0:
                checkpoint_reasons.append("student_actual_eval_count_zero")
            if checkpoint_delta is None or checkpoint_delta >= 0.0:
                checkpoint_reasons.append("student_train_actual_not_improving")
            if validation_delta is None or validation_delta >= 0.0:
                checkpoint_reasons.append("student_validation_actual_not_improving")
            if not operation_nonzero:
                checkpoint_reasons.append("student_operation_missing")
            checkpoint_metrics["student_actual_eval_count"] = actual_eval_count
            checkpoint_metrics["student_actual_gate_ok"] = single_plan_actual_gate_ok
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
                    and single_plan_actual_gate_ok
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
        fixed_validation_plot = plot_fixed_validation_curve(
            metric_csv_paths.get("checkpoint_episode")
        )
        if fixed_validation_plot:
            writer.write(f"FixedValidationPlot: {fixed_validation_plot}")
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
                
        guard_event = apply_actual_compression_guard(
            args=args,
            model=model,
            loss=loss,
            optimizer=optimizer,
            writer=writer,
            guard_state=actual_guard_state,
            checkpoint_metrics=checkpoint_metrics,
            ckpt_dir=ckpt_dir,
            episode=episode,
            runtime_state={
                "optimizer": optimizer,
                "scheduler": scheduler_steplr,
                "scaler": scaler,
                "emulator_optimizer": emulator_optimizer,
                "emulator_scheduler": emulator_scheduler,
                "emulator_scaler": emulator_scaler,
                "mutable_mappings": {
                    "network_k_state_visit_counts": network_k_state_visit_counts,
                },
            },
        )
        autonomy_event = update_network_autonomy_from_guard(args, guard_event)
        if autonomy_event.get("changed") or episode == 0:
            writer.write(
                "NetworkAutonomyUpdate: "
                f"episode={episode + 1}, guard_action={autonomy_event['action']}, "
                f"where_residual_weight={autonomy_event['previous']:.6f}"
                f"->{autonomy_event['current']:.6f}, "
                f"maximum={autonomy_event['maximum']:.6f}"
            )
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


def main():
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


if __name__ == "__main__":
    main()
