import argparse
import os
import sys
from pathlib import Path
from cfgs.utils import str2bool

# sparsepcgc_move_existing_target_only

pretrained_date = "20260524" 
pretrained_time = "230456"

surrogate_date = "20260524"
surrogate_time = "230456"

model_date = "20260524"
model_time = "230456"

# method_com = "OctAttention"
method_com = "SparsePCGC"
# method_com = "G-PCC"
method_name = "Mine"

model_name = "best"

# dataname = "8i"
# dataname = "MVUB"
dataname = "UVG"

# dataset_name = "longdress"
# dataset_name = "loot"
# dataset_name = "redandblack"
# dataset_name = "soldier"

# dataset_name = "andrew"
# dataset_name = "david"
# dataset_name = "phil"
# dataset_name = "ricardo"
# dataset_name = "sarah"

dataset_name = "BlueBackpack"
# dataset_name = "CasualSquat"
# dataset_name = "ElegantDance"
# dataset_name = "Gymnast"
# dataset_name = "ReadyForWinter"

_MASTER_ROOT = Path(__file__).resolve().parents[4]
_DATA_STORAGE_ROOT = (_MASTER_ROOT / "../../../data/maejima").resolve()
_DATA_ROOT = (_DATA_STORAGE_ROOT / "data").resolve()
_PRETRAINED_ROOT = (_DATA_ROOT / "pretrained").resolve()
_LOG_ROOT = (_DATA_STORAGE_ROOT / "log").resolve()
_LEGACY_PRETRAINED_ROOT = (_MASTER_ROOT / "pretrained").resolve()
_DEFAULT_OCTATTENTION_CKPT = (
    _MASTER_ROOT / "compress" / "octree" / "OctAttention" / "modelsave" / "obj" / "encoder_epoch_00800093.pth"
).resolve()
_DEFAULT_SPARSEPCGC_ROOT = (_MASTER_ROOT / "compress" / "octree" / "SparsePCGC").resolve()
_DEFAULT_SPARSEPCGC_CKPT_DENSE = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "dense" / "epoch_last.pth").resolve()
_DEFAULT_SPARSEPCGC_CKPT_DENSE_SR = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "dense_1stage" / "epoch_last.pth").resolve()
_DEFAULT_SPARSEPCGC_CKPT_DENSE_AE = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "dense_slne" / "epoch_last.pth").resolve()
_DEFAULT_SPARSEPCGC_CKPT_SPARSE_LOW = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "sparse_low" / "epoch_last.pth").resolve()
_DEFAULT_SPARSEPCGC_CKPT_SPARSE_HIGH = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "sparse_high" / "epoch_last.pth").resolve()
_DEFAULT_SPARSEPCGC_CKPT_SPARSE_OFFSET = (_DEFAULT_SPARSEPCGC_ROOT / "ckpts" / "sparse_offset" / "epoch_last.pth").resolve()
_DEFAULT_GPCC_ROOT = (_MASTER_ROOT / "compress" / "octree" / "G-PCC").resolve()
_DEFAULT_GPCC_ENCODER = (_DEFAULT_GPCC_ROOT / "build" / "tmc3" / "tmc3").resolve()
_DEFAULT_GPCC_CFG_DIR = (
    _DEFAULT_GPCC_ROOT
    / "cfg"
    / "octree-predlift"
    / "lossless-geom-lossless-attrs"
    / "longdress_vox10_1300"
).resolve()
_DEFAULT_DRACO_ROOT = (_MASTER_ROOT / "compress" / "octree" / "Draco").resolve()
_DEFAULT_DRACO_ENCODER = (_DEFAULT_DRACO_ROOT / "build" / "draco_encoder").resolve()
_DEFAULT_DRACO_DECODER = (_DEFAULT_DRACO_ROOT / "build" / "draco_decoder").resolve()


def _data_subset_dir(split: str, data_name: str = dataname, subset_name: str = dataset_name) -> Path:
    return _DATA_ROOT / split / str(data_name) / str(subset_name)


def _cli_option_was_provided(option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in sys.argv[1:])


def _discover_latest_pretrained_checkpoint(model_stem: str = model_name) -> str:
    candidates = []
    candidates.extend(_LOG_ROOT.glob(f"*/MyNetwork_train/pretrained/*/{model_stem}.pth"))
    candidates.extend(_LOG_ROOT.glob(f"*/MyNetwork_train/checkpoints/*/{model_stem}.pth"))
    candidates.extend(_PRETRAINED_ROOT.glob(f"*/*/{model_stem}.pth"))
    candidates.extend(_LEGACY_PRETRAINED_ROOT.glob(f"*/*/{model_stem}.pth"))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if existing:
        latest = max(existing, key=lambda path: path.stat().st_mtime)
        return str(latest.resolve())
    fallback = _PRETRAINED_ROOT / pretrained_date / f"{pretrained_time}_{method_com}" / f"{model_stem}.pth"
    return str(fallback.resolve())


def _default_checkpoint_path() -> str:
    preferred = _LOG_ROOT / pretrained_date / "MyNetwork_train" / "pretrained" / f"{pretrained_time}_{method_com}" / f"{model_name}.pth"
    if preferred.is_file():
        return str(preferred.resolve())

    lower_method_preferred = _LOG_ROOT / pretrained_date / "MyNetwork_train" / "pretrained" / f"{pretrained_time}_{method_com.lower()}" / f"{model_name}.pth"
    if lower_method_preferred.is_file():
        return str(lower_method_preferred.resolve())

    sparsepcgc_case_preferred = _LOG_ROOT / pretrained_date / "MyNetwork_train" / "pretrained" / f"{pretrained_time}_sparsePCGC" / f"{model_name}.pth"
    if sparsepcgc_case_preferred.is_file():
        return str(sparsepcgc_case_preferred.resolve())

    return str(sparsepcgc_case_preferred.resolve())

def _more_training_checkpoint_path(
    date_value: str,
    time_value: str,
    method_value: str = method_com,
    model_stem: str = model_name,
) -> str:
    # more_training=True のときに、追加学習の初期値として読み込むモデルパスを作る
    path = (
        _LOG_ROOT
        / str(date_value)
        / "MyNetwork_train"
        / "pretrained"
        / f"{time_value}_{method_value}"
        / f"{model_stem}.pth"
    )
    return str(path.resolve())


def _resolve_repo_or_cwd_path(raw_path: str) -> str:
    raw = Path(os.path.expanduser(str(raw_path)))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((_MASTER_ROOT / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1] if candidates else raw)


def _resolve_from_base_path(raw_path: str, base_dir: str) -> str:
    raw = Path(os.path.expanduser(str(raw_path)))
    base = Path(base_dir).expanduser()
    if raw.is_absolute():
        return str(raw.resolve())

    candidates = [
        (Path.cwd() / raw).resolve(),
        (base / raw).resolve(),
        (_MASTER_ROOT / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str((base / raw).resolve())


def _parse_csv_ints(raw_value):
    if isinstance(raw_value, (list, tuple)):
        return [int(value) for value in raw_value]
    text = str(raw_value).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_csv_floats(raw_value):
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    text = str(raw_value).strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _compress_key(raw_value: str) -> str:
    return str(raw_value).strip().lower().replace("_", "").replace("-", "")


def _compress_display_name(raw_value: str) -> str:
    key = _compress_key(raw_value)
    if key == "sparsepcgc":
        return "SparsePCGC"
    if key == "gpcc":
        return "G-PCC"
    if key == "draco":
        return "Draco"
    if key == "octattention":
        return "OctAttention"
    text = str(raw_value).strip()
    return text if text else "OctAttention"


def _pretrained_checkpoint_path(date_value: str, time_value: str, compress_value: str, model_stem: str = model_name) -> str:
    display = _compress_display_name(compress_value)
    raw = str(compress_value).strip()
    run_names = []
    for name in (
        f"{time_value}_{display}",
        f"{time_value}_{raw}",
        f"{time_value}_{raw.lower()}",
        f"{time_value}_sparsePCGC" if _compress_key(compress_value) == "sparsepcgc" else "",
    ):
        if name and name not in run_names:
            run_names.append(name)

    roots = []
    for train_dir in ("Mynetwork_train", "MyNetwork_train"):
        roots.append(_LOG_ROOT / str(date_value) / train_dir)
        roots.append(_LOG_ROOT / str(date_value) / train_dir / "pretrained")

    candidates = [
        root / run_name / f"{model_stem}.pth"
        for root in roots
        for run_name in run_names
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def parse_pugan_args(parser, file_day, file_time):
    """基本情報"""
    parser.add_argument('--date', default=f'{file_day}', type=str, help='日付')
    parser.add_argument('--time', default=f'{file_time}', type=str, help='時刻')
    parser.add_argument('--input_dir', default=str(_DATA_ROOT / "ground"), type=str, help='入力点群データのフォルダパス')
    parser.add_argument('--cpu', action='store_true', help='GPUを使わずCPUで学習するかどうか')
    parser.add_argument('--print_actuator_hard_soft_compare', action='store_true', help='Actuatorのhard/soft出力の統計を取る比較モード')
    parser.add_argument('--print_rate', default=1, type=int, help='ログ出力頻度（1なら毎ステップ、0なら最初と最後のみ）')
    parser.add_argument('--dataname', default=dataname, type=str, help='データセットの名称')
    parser.add_argument('--dataset_name', default=dataset_name, type=str, help='データセット内シーケンスの名称')
    parser.add_argument('--more_training', default=True, type=str2bool, help='学習済みモデルの途中から訓練を再開するか否か')

    parser.add_argument(
        '--repair_operation_amount_logit_weight',
        default=1e-4,
        type=float,
        help='amount head の raw logit を目標操作割合のlogitへ寄せる補助損失重み',
    )
    parser.add_argument(
        '--repair_operation_amount_logit_scale',
        default=6.0,
        type=float,
        help='amount ratio計算時のlogit制限スケール。sigmoid飽和を抑える',
    )
    parser.add_argument(
        '--repair_operation_amount_target_prob_max',
        default=0.98,
        type=float,
        help='amount logit補助教師でtarget/maxが1.0になるときの上限確率。logit目標の過大化を防ぐ',
    )

    """ネットワーク条件"""
    # Network
    parser.add_argument('--encoder_0grad', default=True, type=str2bool, help='Encoderを学習対象にするかどうか')
    parser.add_argument('--prune', default=True, type=str2bool, help='Pruning Moduleを使用するか')
    parser.add_argument('--add', default=True, type=str2bool, help='Adding Moduleを使用するか')
    parser.add_argument('--disp', default=True, type=str2bool, help='Displacement Moduleを使用するか')
    # Encoder, FP Module
    parser.add_argument('--encoder_dim', default=64, type=int, help='各dense blockにおける特徴次元数（入力/出力）')
    parser.add_argument('--out_dim', default=64, type=int, help='各dense blockにおける特徴次元数（入力/出力）')
    parser.add_argument('--local_feat_dim', default=192, type=int, help='局所幾何特徴の次元（Analyzer用）')
    parser.add_argument('--fused_feat_dim', default=64, type=int, help='Encoderで統合された特徴の次元')
    parser.add_argument('--fp_mlp_channels', nargs='+', type=int, default=[128, 64], help='Feature PropagationのMLPの隠れ層チャネル数')
    parser.add_argument('--encoder_pre_downsample', default=True, type=str2bool, help='Encoder入力の前だけcoarse化するか')
    parser.add_argument('--encoder_pre_downsample_mode', default='voxel', type=str, help='Encoder前downsample方法(voxel)')
    parser.add_argument('--encoder_sparse_tensor', default=True, type=str2bool, help='入力点群を点数維持のSparse Tensor表現(量子化座標+occupancy feature)へ変換するか')
    parser.add_argument('--sparse_tensor_keep_after_encoder', default=True, type=str2bool, help='Encoder後も診断/方策決定まではUpsampling後のSparse Tensor経路を維持するか')
    parser.add_argument('--encoder_raw_downsample_factor', default=10.0, type=float, help='Sparse Tensor化後にEncoderへ入れるため何倍ダウンサンプリングするか（10なら点数を約1/10にする）')
    parser.add_argument('--encoder_pre_downsample_max_points', default=8192, type=int, help='Encoderへ入れる最大点数')
    parser.add_argument('--encoder_cdist_max_points', default=4096, type=int, help='pointops CUDAが使えずtorch.cdist KNNへ落ちた時のEncoder最大点数（0なら無効）')
    parser.add_argument('--allow_slow_knn_fallback', default=False, type=str2bool, help='pointops CUDAが使えない時に低速なtorch.cdist fallbackで継続するか')
    parser.add_argument('--encoder_pre_downsample_voxel_scale', default=1.0, type=float, help='qs由来のvoxelサイズの倍率')
    parser.add_argument('--encoder_pre_downsample_growth', default=1.5, type=float, help='voxel数が多すぎるときにvoxelサイズを拡大する倍率')
    parser.add_argument('--encoder_pre_downsample_max_iters', default=8, type=int, help='voxelサイズ調整の最大反復回数')
    parser.add_argument('--encoder_feature_propagation', default='knn_inverse_distance', type=str, help='coarse encoder特徴のfull点群への戻し方')
    parser.add_argument('--encoder_feature_propagation_k', default=3, type=int, help='coarse encoder特徴伝播のkNN数')
    # Pruning Module
    parser.add_argument('--prune_hidden_dim', default=64, type=int, help='Pruning用MLPの隠れ層次元')
    parser.add_argument('--prune_d_high_is_inlier', default=True, type=str2bool, help='dスコアが高いほどインライアとみなすか')
    parser.add_argument('--prune_robust_c', default=2.0, type=float, help='ロバスト重みのパラメータ')
    parser.add_argument('--prune_ratio_min', default=0.85, type=float, help='保持割合の最小値')
    parser.add_argument('--prune_ratio_max', default=0.995, type=float, help='保持割合の最大値')
    parser.add_argument('--prune_use_label_count', default=True, type=str2bool, help='外れ点ラベルの数をPruningの保持点数教師として使うか')
    # Adding Module
    parser.add_argument('--add_hidden_dim', default=64, type=int, help='Add ModuleのMLP隠れ層次元')
    parser.add_argument('--add_fit_ref_max', default=4096, type=int, help='フィッティング損失計算で参照する最大点数')
    parser.add_argument('--add_attr', default=True, type=str2bool, help='追加点に色情報を付与するか')
    parser.add_argument('--add_color', default='Red', type=str, help='追加点の色')
    parser.add_argument('--add_th', default=0.5, type=float, help='追加判定のしきい値')
    parser.add_argument('--target_add_ratio', default=0.003, type=float, help='目標とする追加割合')
    parser.add_argument('--max_add_ratio', default=0.50, type=float, help='追加割合の最大値')
    parser.add_argument('--add_oct_weight', default=1.0, type=float, help='Octreeグリッドへのスナップの重み')
    parser.add_argument('--lambda_sparse', default=1e-3, type=float, help='スパース性の正則化係数')
    # Displacement Module
    parser.add_argument('--disp_hidden_dim', default=64, type=int, help='Displacement用MLPの隠れ層次元')
    parser.add_argument('--max_disp_offset', default=0.002, type=float, help='最大移動距離（メートル）')
    parser.add_argument('--disp_num_blocks',default=4,type=int,help='残差ブロックの数')
    parser.add_argument('--disp_num_steps',default=1,type=int,help='反復更新回数')
    parser.add_argument('--disp_step_size',default=1.0,type=float,help='移動更新のステップサイズ')
    parser.add_argument('--disp_step_decay',default=0.95,type=float,help='ステップサイズの減衰率')
    parser.add_argument('--disp_grad_clip',default=10.0,type=float,help='勾配クリッピング値')
    parser.add_argument('--target_disp_ratio', default=0.25, type=float, help='移動する点の目標割合')
    parser.add_argument('--disp_use_gate',default=True,type=str2bool,help='ゲーティング機構を使うか')
    parser.add_argument('--disp_reg_weight', default=1e-4, type=float, help='移動量の正則化係数')
    parser.add_argument('--disp_ratio_weight', default=1e-4, type=float, help='移動割合の正則化係数')
    parser.add_argument('--disp_occ_weight', default=1e-4, type=float, help='新規Octree voxelを作る変位へのペナルティ係数')
    parser.add_argument('--disp_snap_strength', default=0.35, type=float, help='Octreeグリッドへのスナップ強度')
    parser.add_argument('--disp_use_grid_feat', default=True, type=str2bool, help='Octree量子化位相特徴をDisplacementに入力するか')
    parser.add_argument('--disp_guard_new_voxels', default=True, type=str2bool, help='既存occupied voxel以外へ移動する変位を禁止するか')
    parser.add_argument('--disp_mag_bias', default=-1.0, type=float, help='移動量の初期バイアス')
    parser.add_argument('--disp_gate_bias', default=0.0, type=float, help='ゲートの初期バイアス')
    parser.add_argument('--disp_soft_match_tau', default=0.05, type=float, help='soft top-kの温度パラメータ')


    # Analyzer
    parser.add_argument('--octree_qlevel', type=int, default=12,help='Octree量子化レベル')
    parser.add_argument('--octree_ctx_level', type=int, default=5,help='Octreeコンテキストの深さ')
    parser.add_argument('--octree_ctx_dim', type=int, default=8,help='Octreeコンテキスト特徴次元')
    parser.add_argument('--outlier_label_th_scale', default=4.0, type=float, help='外れ値判定のしきい値スケール（MAD基準）')
    parser.add_argument('--outlier_label_min_ratio', default=0.03, type=float, help='外れ値割合の最小値')
    parser.add_argument('--outlier_label_max_ratio', default=0.15, type=float, help='外れ値割合の最大値')
    # Cause-decomposed octree structure repair network
    parser.add_argument('--structure_hidden_dim', default=96, type=int, help='原因分解・修復ポリシーMLPの隠れ次元')
    parser.add_argument('--repair_actuator_hidden_dim', default=64, type=int, help='構造修復アクチュエータの隠れ次元')
    parser.add_argument('--repair_policy_temperature', default=1.0, type=float, help='修復プリミティブsoftmax温度')
    parser.add_argument('--repair_policy_entropy_weight', default=1e-3, type=float, help='修復ポリシーのエントロピー正則化')
    parser.add_argument('--max_repair_offset', default=0.002, type=float, help='構造修復で許す最大移動距離')
    parser.add_argument('--max_repair_qstep', default=0.20, type=float, help='構造修復で許す最大移動距離を量子化step比で指定する下限')
    parser.add_argument('--repair_snap_strength', default=0.20, type=float, help='Octree格子中心へ寄せる強さ')
    parser.add_argument('--target_repair_ratio', default=0.15, type=float, help='修復対象点の目標割合')
    parser.add_argument('--max_repair_ratio', default=1.00, type=float, help='操作量学習時に候補として残す修復対象点の最大割合')
    parser.add_argument('--repair_ratio_weight', default=8.0, type=float, help='修復割合制御損失の重み')
    parser.add_argument('--repair_shape_guard_weight', default=0.5, type=float, help='形状保持原因が強い点を動かしすぎない正則化')
    parser.add_argument('--target_drop_ratio', default=0.02, type=float, help='点削除ゲートの目標割合')
    parser.add_argument('--max_drop_ratio', default=0.50, type=float, help='点削除ゲートの上限割合')
    parser.add_argument('--repair_delete_max_points_per_voxel', default=8, type=int, help='削除候補にするleaf voxel内の最大点数(0なら無制限)')
    parser.add_argument('--repair_move_max_points_per_voxel', default=8, type=int, help='move source候補にするleaf voxel内の最大点数(0なら無制限)')
    parser.add_argument(
        '--repair_move_relax_voxel_count_when_starved',
        default=True,
        type=str2bool,
        help='Adjust候補が少なすぎる場合だけrepair_move_max_points_per_voxel制限を緩める',
    )
    parser.add_argument(
        '--repair_move_candidate_min_ratio',
        default=0.05,
        type=float,
        help='Adjust候補不足とみなす最小候補割合',
    )
    parser.add_argument('--repair_drop_ratio_weight', default=4.0, type=float, help='点削除割合制御損失の重み')
    parser.add_argument('--repair_drop_shape_guard_weight', default=1.0, type=float, help='形状保持点を削除しすぎない正則化')
    parser.add_argument('--repair_drop_soft_proxy_tau', default=8.0, type=float, help='Prune soft proxyでdrop logit飽和を避けるための温度')
    parser.add_argument('--repair_drop_direct_target_weight', default=5.0, type=float, help='drop soft proxyを目標削除率へ近づける正則化重み')
    parser.add_argument('--repair_drop_entropy_weight', default=0.01, type=float, help='drop soft proxyのエントロピー正則化重み')
    parser.add_argument('--repair_add_ratio_weight', default=4.0, type=float, help='点追加割合制御損失の重み')
    parser.add_argument('--repair_add_shape_guard_weight', default=0.5, type=float, help='形状保持点の近傍に追加しすぎない正則化')
    parser.add_argument('--repair_add_offset_weight', default=0.25, type=float, help='追加点オフセットの正則化')
    parser.add_argument('--repair_add_drop_conflict_weight', default=2.0, type=float, help='削除される点を基点に追加する無駄操作へのペナルティ')
    parser.add_argument('--repair_add_keep_weight', default=1.0, type=float, help='追加した点が推論時hardeningで消えないようにする損失重み')
    parser.add_argument('--repair_add_min_offset_qstep', default=0.20, type=float, help='追加点が基点と重複しないための最小オフセット(量子化step比)')
    parser.add_argument('--repair_add_min_offset_weight', default=0.5, type=float, help='追加点オフセットが小さすぎる時のペナルティ重み')
    parser.add_argument('--repair_move_require_empty_target', default=True, type=str2bool, help='移動先を空の近傍量子化ボクセルに制限するか')
    parser.add_argument('--repair_move_prefer_occupied_target', default=False, type=str2bool, help='移動先候補に既存occupied voxelを優先し、codec上のmergeを促すか')
    parser.add_argument('--repair_move_source_prior_weight', default=0.35, type=float, help='原因診断scoreからmove source候補を起こす補助重み')
    parser.add_argument('--repair_selection_mode', default='target', type=str, help='修復操作のhard選択方式(target/threshold_cap)。threshold_capでは目標割合を強制せず上限として扱う')
    parser.add_argument('--repair_move_hard_threshold', default=0.5, type=float, help='threshold_cap時に移動をhard化する最小score')
    parser.add_argument('--target_move_ratio', default=0.05, type=float, help='Adjust/Moveの目標実行割合。未指定時もtarget_repair_ratioとは分けて扱う')
    parser.add_argument('--max_move_ratio', default=1.00, type=float, help='Adjustの学習済み実行割合の上限')
    parser.add_argument(
        '--repair_move_ratio_floor',
        default=0.03,
        type=float,
        help='学習中にAdjustのHard実行割合が0へ潰れないようにする最小forward割合。backwardは元のlearned ratioへ流す',
    )
    parser.add_argument('--repair_move_warmup_steps', default=0, type=int, help='Adjust/Move実行上限を学習初期に徐々に上げるstep数')
    parser.add_argument('--repair_drop_hard_threshold', default=0.5, type=float, help='threshold_cap時に削除をhard化する最小score')
    parser.add_argument('--repair_add_hard_threshold', default=0.5, type=float, help='threshold_cap時に追加をhard化する最小score')
    parser.add_argument('--repair_quant_guard_weight', default=1.0, type=float, help='量子化ボクセル上で無効な移動/追加を抑える正則化')
    parser.add_argument('--repair_local_guard_weight', default=0.25, type=float, help='形状保持原因が強い局所点の削除/移動を抑える正則化')
    parser.add_argument('--add_noop_keep_threshold', default=0.5, type=float, help='このkeep確率未満の点は追加基点から除外する')
    parser.add_argument('--repair_add_weight_mode', default='hard', type=str, help='追加点のfinal_wをhard/softのどちらで作るか')
    parser.add_argument('--repair_exploration_fraction', default=0.0, type=float, help='全学習stepのうちadd/drop探索ノイズを残す割合')
    parser.add_argument('--repair_add_candidate_ratio_start', default=0.0, type=float, help='探索初期の追加候補割合(0ならmax_add_ratio)')
    parser.add_argument('--repair_add_candidate_ratio_end', default=0.0, type=float, help='探索終了後の追加候補割合(0ならmax_add_ratio)')
    parser.add_argument('--repair_add_score_noise_start', default=0.0, type=float, help='探索初期に追加位置logitへ入れるGumbelノイズ量')
    parser.add_argument('--repair_add_score_noise_end', default=0.0, type=float, help='探索終了後に追加位置logitへ入れるGumbelノイズ量')
    parser.add_argument('--repair_add_weight_random_mix_start', default=0.0, type=float, help='探索初期に追加点final_wへ混ぜるランダム重み割合')
    parser.add_argument('--repair_add_weight_random_mix_end', default=0.0, type=float, help='探索終了後に追加点final_wへ混ぜるランダム重み割合')
    parser.add_argument('--repair_learn_operation_amounts', default=True, type=str2bool, help='Add/Adjustの実行量を固定比率ではなくActuator特徴から学習する')
    parser.add_argument('--repair_operation_amount_bias_scale', default=2.0, type=float, help='学習されたAdd/Adjust実行量を位置logitへ反映する強さ')
  
    """損失項の重みパラメータ"""
    # 圧縮損失における点操作のAmount
    parser.add_argument('--repair_amount_downstream_grad_scale', default=10.0, type=float, help='Amount ratioから実際の点操作へ向かう下流勾配だけを強める倍率。forward値は変えず、backwardだけ強める')
    parser.add_argument('--repair_drop_amount_downstream_grad_scale', default=20.0, type=float, help='Prune Amount ratioから実際の削除操作へ向かう下流勾配だけを強める倍率')
    parser.add_argument('--repair_add_amount_downstream_grad_scale', default=10, type=float, help='Add Amount ratioから実際の追加操作へ向かう下流勾配だけを強める倍率')
    parser.add_argument('--repair_move_amount_downstream_grad_scale', default=8.0, type=float, help='Move Amount ratioから実際の移動操作へ向かう下流勾配だけを強める倍率')
    parser.add_argument('--repair_amount_downstream_grad_max_scale', default=8.0, type=float, help='Amount downstream STE倍率の上限。極端な勾配増幅を抑える')
    parser.add_argument('--repair_soft_normalizer_floor', default=1e-4, type=float, help='Soft操作量正規化の分母下限。AMP fp16で0除算/inf勾配を防ぐ')
    # 圧縮損失における点操作のWhere
    parser.add_argument('--repair_where_downstream_grad_scale', default=1.0, type=float, help='Where scoreから実際の点操作へ向かう下流勾配だけを強める倍率。forward値は変えず、backwardだけ強める')
    parser.add_argument('--repair_drop_where_downstream_grad_scale', default=1.0, type=float, help='Prune Where scoreから実際の削除操作へ向かう下流勾配だけを強める倍率')
    parser.add_argument('--repair_add_where_downstream_grad_scale', default=1.0, type=float, help='Add Where scoreから実際の追加操作へ向かう下流勾配だけを強める倍率')
    parser.add_argument('--repair_move_where_downstream_grad_scale', default=2.0, type=float, help='Move Where scoreから実際の移動操作へ向かう下流勾配だけを強める倍率')
    # Actuatorにおける点操作のWhere
    parser.add_argument('--repair_drop_where_actuator_weight', default=0.3, type=float, help='Prune Where scoreへ直接かけるActuator補助損失重み')
    parser.add_argument('--repair_add_where_actuator_weight', default=0.3, type=float, help='Add Where scoreへ直接かけるActuator補助損失重み')
    parser.add_argument('--repair_move_where_actuator_weight', default=0.3, type=float, help='Move Where scoreまたは方向へ直接かけるActuator補助損失重み')
    # Actuatorにおける点操作のAmount
    parser.add_argument('--repair_operation_amount_consistency_weight', default=0.1, type=float, help='学習済み操作割合と実soft操作率を一致させる補助損失重み')
    parser.add_argument('--repair_operation_amount_direct_weight', default=0.01, type=float, help='learned操作割合を目標割合へ近づける直接補助損失。圧縮主目的では弱く使う')
    parser.add_argument('--repair_drop_amount_supervision_weight', default=0.001, type=float, help='Prune hard実行量を教師にするAmount補助損失。圧縮主目的では弱く使う')
    parser.add_argument('--repair_drop_amount_soft_consistency_weight', default=0.0005, type=float, help='Prune soft実行量とlearned ratioの整合補助損失')
    parser.add_argument('--repair_move_amount_supervision_weight', default=0.001, type=float, help='Move hard実行量を教師にするAmount補助損失。圧縮主目的では弱く使う')
    parser.add_argument('--repair_move_amount_soft_consistency_weight', default=0.0005, type=float, help='Move soft実行量とlearned ratioの整合補助損失')
    parser.add_argument('--repair_add_amount_supervision_weight', default=0.001, type=float, help='Add hard実行量を教師にするAmount補助損失。圧縮主目的では弱く使う')
    parser.add_argument('--repair_add_amount_soft_consistency_weight', default=0.0005, type=float, help='Add soft実行量とlearned ratioの整合補助損失')

    parser.add_argument('--repair_drop_amount_random_mix_start', default=0.1, type=float, help='探索初期にPrune実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_drop_amount_random_mix_end', default=0.0, type=float, help='探索終了後にPrune実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_add_amount_random_mix_start', default=0.1, type=float, help='探索初期にAdd実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_add_amount_random_mix_end', default=0.0, type=float, help='探索終了後にAdd実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_move_amount_random_mix_start', default=0.1, type=float, help='探索初期にAdjust実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_move_amount_random_mix_end', default=0.0, type=float, help='探索終了後にAdjust実行量へ混ぜるランダム割合')
    parser.add_argument('--repair_move_score_noise_start', default=0.0, type=float, help='探索初期にAdjust source scoreへ入れる正規ノイズ量')
    parser.add_argument('--repair_move_score_noise_end', default=0.0, type=float, help='探索終了後にAdjust source scoreへ入れる正規ノイズ量')
    parser.add_argument('--repair_drop_score_noise_start', default=0.0, type=float, help='探索初期に削除logitへ入れる正規ノイズ量')
    parser.add_argument('--repair_drop_score_noise_end', default=0.0, type=float, help='探索終了後に削除logitへ入れる正規ノイズ量')
    parser.add_argument('--repair_drop_random_mix_start', default=0.0, type=float, help='探索初期に削除確率へ混ぜるランダム削除マスク割合')
    parser.add_argument('--repair_drop_random_mix_end', default=0.0, type=float, help='探索終了後に削除確率へ混ぜるランダム削除マスク割合')
    parser.add_argument('--repair_priority_gate', default=True, type=str2bool, help='高コスト領域だけを修復対象にする優先度ゲートを使うか')
    parser.add_argument('--repair_priority_gate_tau', default=0.08, type=float, help='修復優先度ゲートの温度')
    parser.add_argument('--repair_gate_mean_cap', default=True, type=str2bool, help='修復対象割合の平均がtarget_repair_ratioを超えないように再スケールするか')
    parser.add_argument('--repair_unit_level', default=5, type=int, help='原因集約で使う粗いOctree/subtree単位の深さ')
    parser.add_argument('--allow_local_repair_unit_recompute', default=False, type=str2bool, help='unit_keysが無い場合にCauseDiagnosisAggregation内で局所Repair Unitを再計算するか')
    parser.add_argument('--allow_local_octree_recompute', default=False, type=str2bool, help='prebuilt_subtree_tree以外のdebug用途でOctreeStructureAnalysisの局所Octree再計算を許可するか')
    parser.add_argument('--forbid_local_voxel_recompute', default=False, type=str2bool, help='StructureRepairActuatorでoctree_context無しの局所Voxel再計算を禁止するか')
    parser.add_argument('--structure_geo_k', default=8, type=int, help='構造解析に使う局所幾何kNN数')
    parser.add_argument('--structure_geo_max_points', default=2048, type=int, help='局所幾何統計を厳密計算する最大点数（超過時はOctree特徴を優先）')
    parser.add_argument('--octree_diag_levels', default='4,6,8,10,12', type=str, help='Octree階層ごとの診断ログを出すレベル')
    parser.add_argument('--training_stage', default='joint', type=str, help='学習段階(diagnosis/joint)')
    parser.add_argument('--two_stage_training', default=False, type=str2bool, help='2段階学習(diagnosis->joint)を自動で行うか')
    parser.add_argument('--diagnosis_episode_ratio', default=0.25, type=float, help='全episodeのうちdiagnosis段階に使う割合')
    parser.add_argument('--diagnosis_episodes', default=0, type=int, help='diagnosis段階のepisode数(0ならratioから自動計算)')
    parser.add_argument('--diagnosis_actuator_strength', default=0.1, type=float, help='diagnosis段階でのactuator強度')
    parser.add_argument('--repair_actuator_strength', default=0.50, type=float, help='joint段階でのactuator強度')
    parser.add_argument('--diagnosis_geom_factor', default=0.1, type=float, help='diagnosis段階で幾何損失に掛ける係数')
    parser.add_argument('--diagnosis_com_factor', default=0.25, type=float, help='diagnosis段階で圧縮損失に掛ける係数')
    parser.add_argument('--diagnosis_attr_factor', default=1.0, type=float, help='diagnosis段階で原因分解損失に掛ける係数')
    parser.add_argument('--diagnosis_policy_factor', default=1.0, type=float, help='diagnosis段階で方策整合損失に掛ける係数')
    parser.add_argument('--diagnosis_repair_factor', default=0.25, type=float, help='diagnosis段階で内部正則化損失に掛ける係数')
    parser.add_argument('--attr_node_weight', default=1.0, type=float, help='node count原因教師の重み')
    parser.add_argument('--attr_single_weight', default=1.5, type=float, help='single-child chain原因教師の重み')
    parser.add_argument('--attr_lowprob_weight', default=1.5, type=float, help='低確率occupancy原因教師の重み')
    parser.add_argument('--attr_context_weight', default=1.0, type=float, help='context difficulty原因教師の重み')
    parser.add_argument('--attr_quant_weight', default=1.25, type=float, help='量子化ボクセル無駄原因教師の重み')
    parser.add_argument('--attr_sparse_weight', default=1.0, type=float, help='sparse fragmentation原因教師の重み')
    parser.add_argument('--attr_outlier_weight', default=1.0, type=float, help='outlier原因教師の重み')
    parser.add_argument('--attr_shape_weight', default=0.75, type=float, help='形状保持原因教師の重み')
    parser.add_argument('--loss_attr_scale', default=0.05, type=float, help='原因分解損失の内部スケール')
    parser.add_argument('--loss_policy_scale', default=1.0, type=float, help='修復ポリシー損失の内部スケール')
    parser.add_argument('--loss_repair_scale', default=1.0, type=float, help='修復アクチュエータ損失の内部スケール')
    # その他設定
    parser.add_argument('--encoder_bn', default=False, type=str2bool, help='EncoderでBatchNormを使うか')
    parser.add_argument('--k', default=8, type=int, help='近傍点数（kNN）')
    parser.add_argument('--global_mlp', default=True, type=str2bool, help='global MLPを使うか')
    parser.add_argument('--encoder_query_chunk', default=2048, type=int, help='Encoderのattention計算の分割サイズ（0で無効）')

    """Train"""
    parser.add_argument('--save_dir', default=str((_DATA_ROOT / 'trained_model').resolve()), type=str, help='モデル保存ディレクトリ')
    parser.add_argument('--ckpt', default=_default_checkpoint_path(), type=str, help='チェックポイントのパス')
    parser.add_argument('--pretrained_date', default=pretrained_date, type=str, help='test.pyで読む学習済みモデルの日付')
    parser.add_argument('--pretrained_time', default=pretrained_time, type=str, help='test.pyで読む学習済みモデルの時刻')
    parser.add_argument(
        '--more_training_ckpt',
        default='',
        type=str,
        help='more_training=True のときに追加学習の初期値として読み込むモデルパス。空なら pretrained_date/pretrained_time/method_com/model_name から自動生成する',
    )
    parser.add_argument('--out_path', default=str((_LOG_ROOT / file_day / "MyNetwork_train" / "pretrained" / file_time).resolve()), type=str, help='チェックポイント保存先')
    parser.add_argument('--log_root', default=str(_LOG_ROOT), type=str, help='学習・推論ログ保存ルート')
    parser.add_argument('--optim', default='adam', type=str, help='最適化手法（adamまたはsgd）')
    parser.add_argument('--expansion', action='store_true', help='拡張データを使用するか')
    parser.add_argument('--gamma', default=0.5, type=float, help='学習率減衰の係数')
    parser.add_argument('--lr_decay_step', default=24, type=int, help='学習率を減衰させるステップ間隔')
    parser.add_argument('--lr_scheduler_enabled', default=False, type=str2bool, help='TrueならEpoch単位のStepLRを使う。SparsePCGCではLR崩壊防止のため既定でFalse')
    parser.add_argument('--min_main_lr', default=1e-5, type=float, help='main optimizerの学習率floor')
    parser.add_argument('--min_surrogate_lr', default=1e-6, type=float, help='Surrogate optimizerの学習率floor')
    parser.add_argument('--max_files', default=30, type=int, help='読み込む最大ファイル数')
    parser.add_argument('--episodes', default=128, type=int, help='学習エピソード数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学習率')
    parser.add_argument('--save_eval', default='loss', type=str, help='評価指標（lossまたはpsnr）')
    parser.add_argument('--deform', default=False, type=str2bool, help='変形モジュールをゆっくり学習するか')
    parser.add_argument('--loss_type', default='cd', type=str, help='幾何損失の種類')
    parser.add_argument('--method_name', default=method_name, type=str, help='ログ上の提案手法名')
    parser.add_argument('--run_name', default='', type=str, help='チェックポイント保存名。空なら <time>_<compress> を使う')
    parser.add_argument('--geometry_audit_max_points', default=8192, type=int, help='geometry監査用にCDを計算する最大点数(0で無効)')
    parser.add_argument('--operation_count_drop_threshold', default=0.50, type=float, help='学習ログで削除点として数えるkeep確率のしきい値')
    parser.add_argument('--operation_count_adjust_threshold', default=1e-6, type=float, help='学習ログで調整点として数える最小移動距離')

    parser.add_argument(
        '--loss_mode',
        default='compression_primary',
        choices=['legacy_total', 'compression_primary'],
        type=str,
        help='loss構成。legacy_totalは既存互換、compression_primaryは圧縮主目的の実験モード',
    )
    parser.add_argument('--compression_primary_warmup_steps', default=0, type=int, help='compression_primaryでw_comを滑らかに上げるstep数(0で無効)')
    parser.add_argument('--cp_lambda_geom', default=1.0, type=float, help='compression_primaryのgeometry safety penalty重み')
    parser.add_argument('--cp_lambda_single', default=0.05, type=float, help='compression_primaryのsingle-child safety penalty重み')
    parser.add_argument('--cp_lambda_nodes', default=0.03, type=float, help='compression_primaryのnode safety penalty重み')
    parser.add_argument('--cp_lambda_actuator', default=0.05, type=float, help='compression_primaryのactuator safety penalty重み')
    parser.add_argument('--cp_lambda_sparsepcgc', default=0.0, type=float, help='compression_primaryのSparsePCGC soft aux safety penalty重み')
    parser.add_argument('--cp_lambda_op', default=0.5, type=float, help='compression_primaryのoperation safety penalty重み(現状は実験用、default無効)')
    parser.add_argument(
        '--compression_primary_proxy_grad_weight',
        default=0.10,
        type=float,
        help='compression_primaryでhard圧縮目的のforward値を保ったまま、backwardだけ微分可能proxyへ流す重み',
    )
    parser.add_argument('--cp_tau_geom', default=0.06, type=float, help='compression_primaryのgeometry許容閾値')
    parser.add_argument('--cp_tau_single', default=0.0, type=float, help='compression_primaryのsingle-child許容閾値')
    parser.add_argument('--cp_tau_nodes', default=0.0, type=float, help='compression_primaryのnode許容閾値')
    parser.add_argument('--cp_tau_actuator', default=0.0, type=float, help='compression_primaryのactuator許容閾値')
    parser.add_argument('--cp_tau_sparsepcgc', default=0.0, type=float, help='compression_primaryのSparsePCGC soft aux許容閾値')
    parser.add_argument('--cp_use_stage_factors', default=False, type=str2bool, help='compression_primaryでも既存stage factorを使うか(default False)')
    parser.add_argument('--cp_force_joint_actuator', default=True, type=str2bool, help='compression_primaryでactuator strengthをjoint相当に固定するか')
    parser.add_argument('--cp_log_grad_terms', default=True, type=str2bool, help='compression_primaryの各loss項のrequires_grad/finiteをログするか')
    parser.add_argument('--w_geom',     default=10**4, type=float, help='幾何損失ブロック全体の重み')
    parser.add_argument('--geom_d2_weight', default=0.2, type=float, help='loss_type=cd+d2 のときの D2PSNR 報酬項の重み')
    parser.add_argument('--w_com',      default=1, type=float, help='圧縮損失ブロック全体の重み（actual/surrogate backendでは total-bit 差[%%] に直接掛かる）')
    parser.add_argument('--w_attr',     default=1, type=float, help='原因分解損失ブロック全体の重み')
    parser.add_argument('--w_policy',   default=0.03, type=float, help='構造修復ポリシー損失ブロック全体の重み')
    parser.add_argument('--w_actuator', default=7, type=float, help='構造修復アクチュエータ正則化損失ブロック全体の重み')
    parser.add_argument('--w_prun',     default=None, type=float, help='旧名: --w_attr の後方互換alias')
    parser.add_argument('--w_add',      default=None, type=float, help='旧名: --w_policy の後方互換alias')
    parser.add_argument('--w_dis',      default=None, type=float, help='旧名: --w_actuator の後方互換alias')
    parser.add_argument('--com_bit',    default=10*100, type=float, help='proxy backend で bit 差(%%)項へ掛ける重み')
    parser.add_argument('--com_sin',    default=1, type=float, help='proxy backend で single-child 差(%%)項へ掛ける重み')
    parser.add_argument('--com_node',   default=4, type=float, help='proxy backend で node 数差(%%)項へ掛ける重み')
    parser.add_argument('--com_bpn',    default=0.25, type=float, help='proxy backend で bits-per-node 差(%%)項へ掛ける重み')
    parser.add_argument('--actual_total_bit_objective_mix', default=1.0, type=float, help='actual/surrogate backendでL_com直結と内訳合成を混ぜる比率。1.0なら実bit差分のみ')
    parser.add_argument('--com_sparsepcgc', default=0.0, type=float, help='SparsePCGC補助proxy項へ掛ける重み。主objectiveには既定で混ぜない')
    parser.add_argument('--com_lowprob', default=1, type=float, help='proxy backend で low-probability occupancy 項へ掛ける重み')
    parser.add_argument('--com_ent',   default=2, type=float, help='旧 proxyOctreeCompression 経路のエントロピー項重み（現行構造修復lossでは未使用）')
    parser.add_argument('--prun_cnt',   default=5, type=float, help='旧 Pruning loss の個数制御重み（現行構造修復lossでは未使用）')
    parser.add_argument('--prun_out',   default=20*100, type=float, help='旧 Pruning loss の外れ値重み（現行構造修復lossでは未使用）')
    parser.add_argument('--add_cnt',    default=5, type=float, help='旧 Add loss の個数制御重み（現行構造修復lossでは未使用）')
    parser.add_argument('--add_fit',    default=4*100, type=float, help='旧 Add loss のフィッティング重み（現行構造修復lossでは未使用）')
    parser.add_argument('--add_rep',    default=1*100, type=float, help='旧 Add loss の分散抑制重み（現行構造修復lossでは未使用）')
    parser.add_argument('--disp_cnt',    default=5, type=float, help='旧 Displacement loss の個数制御重み（現行構造修復lossでは未使用）')
    parser.add_argument('--disp_fit',    default=4*100, type=float, help='旧 Displacement loss のフィッティング重み（現行構造修復lossでは未使用）')

    parser.add_argument('--lambda_p',   default=10**-5, type=float, help='soft圧縮損失の係数')
    parser.add_argument('--discrete_loss_mode', default='ste_hard', type=str, help='離散学習のモード')
    parser.add_argument('--discrete_surrogate_weight', default=1.0, type=float, help='STE時の代理勾配の重み')
    parser.add_argument('--discrete_policy_weight', default=1, type=float, help='ポリシー勾配の重み')
    parser.add_argument('--discrete_policy_reward_clip', default=100.0, type=float, help='報酬のクリップ値（0で無効）')
    parser.add_argument('--discrete_policy_baseline_momentum', default=0.95, type=float, help='ベースラインのEMA係数')

    """Compression"""
    parser.add_argument('--compress', default='SparsePCGC', type=str, help='使用する圧縮手法')
    parser.add_argument('--octree_voxel', type=float, default=1e-3, help='Octreeボクセルサイズ')
    parser.add_argument('--qs', type=int, default=2, help='量子化ステップサイズ')

    # Octree Compression
    parser.add_argument('--max_gpu_mem_it', type=int, default=2**9, help='GPUメモリ制限に応じた反復回数')
    parser.add_argument('--oa_subprocess', default=False, type=str2bool, help='サブプロセスで圧縮を行うか')
    parser.add_argument('--surrogate', default=True, type=str2bool, help='TrueならproxyではなくOctAttention surrogateを圧縮損失に使う')
    parser.add_argument('--compression_loss_backend', default='proxy', type=str, help='圧縮損失の計算方法(proxy/octattention_actual/octattention_actual_ste/octattention_surrogate/sparsepcgc_actual/sparsepcgc_actual_ste/sparsepcgc_surrogate/gpcc_actual/gpcc_actual_ste/gpcc_surrogate/draco_actual/draco_actual_ste/draco_surrogate)。surrogateは実圧縮教師の百分率を周期的に模倣する')
    parser.add_argument('--compression_grad_probe', default=False, type=str2bool, help='圧縮損失から出力点群へ勾配が流れるか各stepで表示するか')
    parser.add_argument('--compression_grad_probe_every', default=10, type=int, help='圧縮損失の勾配診断を何回に1回表示するか')
    parser.add_argument('--octattention_actualcode', default=False, type=str2bool, help='OctAttention実圧縮で算術符号化後の実bitを使うか（学習中はFalse推奨）')
    parser.add_argument('--octattention_ckpt', default=str(_DEFAULT_OCTATTENTION_CKPT), type=str, help='OctAttention encoder checkpoint')
    parser.add_argument('--octattention_tmp_dir', default='', type=str, help='OctAttention実圧縮用の一時ディレクトリ（空なら/dev/shm優先）')
    parser.add_argument('--sparsepcgc_env', default='sparsepcgc', type=str, help='SparsePCGC teacherを実行するconda環境名')
    parser.add_argument('--sparsepcgc_python', default='', type=str, help='SparsePCGC teacher用Pythonの絶対パス（空ならsparsepcgc_envから探索）')
    parser.add_argument('--sparsepcgc_root', default=str(_DEFAULT_SPARSEPCGC_ROOT), type=str, help='SparsePCGCリポジトリのパス')
    parser.add_argument('--sparsepcgc_mode', default='dense_lossless', type=str, help='SparsePCGC mode(dense_lossless/dense_lossy/sparse_lossless/sparse_lossy_gpcc)')
    parser.add_argument('--sparsepcgc_device', default='auto', type=str, help='SparsePCGC teacherの実行先(auto/cuda/cpu/cuda:0など)')
    parser.add_argument('--sparsepcgc_tmp_dir', default='', type=str, help='SparsePCGC teacher用一時ディレクトリ（空なら/dev/shm優先）')
    parser.add_argument('--sparsepcgc_timeout', default=600.0, type=float, help='SparsePCGC teacherの1リクエスト待ち時間（秒）')
    parser.add_argument('--sparsepcgc_skip_decode', default=True, type=str2bool, help='SparsePCGC teacherで復号を省略してbitだけ計測するか')
    parser.add_argument('--sparsepcgc_ckptdir', default=str(_DEFAULT_SPARSEPCGC_CKPT_DENSE), type=str, help='SparsePCGC dense checkpoint')
    parser.add_argument('--sparsepcgc_ckptdir_sr', default=str(_DEFAULT_SPARSEPCGC_CKPT_DENSE_SR), type=str, help='SparsePCGC dense SR checkpoint')
    parser.add_argument('--sparsepcgc_ckptdir_ae', default=str(_DEFAULT_SPARSEPCGC_CKPT_DENSE_AE), type=str, help='SparsePCGC dense AE checkpoint')
    parser.add_argument('--sparsepcgc_ckptdir_low', default=str(_DEFAULT_SPARSEPCGC_CKPT_SPARSE_LOW), type=str, help='SparsePCGC sparse low checkpoint')
    parser.add_argument('--sparsepcgc_ckptdir_high', default=str(_DEFAULT_SPARSEPCGC_CKPT_SPARSE_HIGH), type=str, help='SparsePCGC sparse high checkpoint')
    parser.add_argument('--sparsepcgc_ckptdir_offset', default=str(_DEFAULT_SPARSEPCGC_CKPT_SPARSE_OFFSET), type=str, help='SparsePCGC offset checkpoint')
    parser.add_argument('--sparsepcgc_offset', default=False, type=str2bool, help='SparsePCGC sparse_lossy_gpccでoffset modelを使うか')
    parser.add_argument('--sparsepcgc_match_qs', default=False, type=str2bool, help='SparsePCGCの有効量子化幅を--qsに合わせる（SparsePCGC本体条件を優先するならFalse）')
    parser.add_argument('--sparsepcgc_voxel_size', default=1.0, type=float, help='SparsePCGC load_sparse_tensorのvoxel_size')
    parser.add_argument('--sparsepcgc_pos_quantscale', default=1, type=int, help='SparsePCGC posQuantscale')
    parser.add_argument('--sparsepcgc_psnr_resolution', default=1023, type=int, help='SparsePCGC lossy評価用PSNR resolution')
    parser.add_argument('--sparsepcgc_test_d2', default=False, type=str2bool, help='SparsePCGC lossy評価でD2を計算するか')
    parser.add_argument('--sparsepcgc_dense_scale_ae_list', default='1,0,1,0,1,0', type=str, help='SparsePCGC dense_lossy用AE scale list')
    parser.add_argument('--sparsepcgc_dense_scale_sr_list', default='0,1,1,2,2,3', type=str, help='SparsePCGC dense_lossy用SR scale list')
    parser.add_argument('--sparsepcgc_pos_quantscale_list', default='4', type=str, help='SparsePCGC sparse_lossy_gpcc用posQuantscale list')
    parser.add_argument('--gpcc_root', default=str(_DEFAULT_GPCC_ROOT), type=str, help='G-PCCリポジトリのパス')
    parser.add_argument('--gpcc_encoder_path', default=str(_DEFAULT_GPCC_ENCODER), type=str, help='G-PCC tmc3 encoderのパス')
    parser.add_argument('--gpcc_cfg_dir', default=str(_DEFAULT_GPCC_CFG_DIR), type=str, help='G-PCC encoder.cfgを含む設定ディレクトリ')
    parser.add_argument('--gpcc_tmp_dir', default='', type=str, help='G-PCC teacher用一時ディレクトリ（空なら/dev/shm優先）')
    parser.add_argument('--gpcc_timeout', default=120.0, type=float, help='G-PCC teacherの1リクエスト待ち時間（秒）')
    parser.add_argument('--gpcc_match_qs', default=True, type=str2bool, help='G-PCC teacher前の有効量子化幅を--qsに合わせる（明示指定が無い場合）')
    parser.add_argument('--gpcc_prequantize', default=True, type=str2bool, help='tmc3入力前に点群を整数Octree座標へ量子化する')
    parser.add_argument('--gpcc_effective_qs', default=1.0, type=float, help='G-PCC teacher前量子化に使う有効量子化幅')
    parser.add_argument('--gpcc_disable_attribute_coding', default=True, type=str2bool, help='G-PCC teacherでは幾何のみを符号化する')
    parser.add_argument('--gpcc_merge_duplicated_points', default=True, type=str2bool, help='G-PCC teacherで重複点を統合する')
    parser.add_argument('--draco_root', default=str(_DEFAULT_DRACO_ROOT), type=str, help='Dracoリポジトリのパス')
    parser.add_argument('--draco_encoder_path', default=str(_DEFAULT_DRACO_ENCODER), type=str, help='Draco encoderバイナリのパス')
    parser.add_argument('--draco_decoder_path', default=str(_DEFAULT_DRACO_DECODER), type=str, help='Draco decoderバイナリのパス')
    parser.add_argument('--draco_tmp_dir', default='', type=str, help='Draco teacher用一時ディレクトリ（空なら/dev/shm優先）')
    parser.add_argument('--draco_timeout', default=120.0, type=float, help='Draco teacherの1リクエスト待ち時間（秒）')
    parser.add_argument('--draco_match_qs', default=True, type=str2bool, help='Draco teacher前の有効量子化幅を--qsに合わせる（明示指定が無い場合）')
    parser.add_argument('--draco_prequantize', default=True, type=str2bool, help='Draco入力前に点群を整数格子へ量子化する')
    parser.add_argument('--draco_effective_qs', default=1.0, type=float, help='Draco teacher前量子化に使う有効量子化幅')
    parser.add_argument('--draco_position_quantization_bits', default=0, type=int, help='Draco -qp。prequantize=Trueでは0推奨')
    parser.add_argument('--draco_compression_level', default=7, type=int, help='Draco -cl [0-10]')
    parser.add_argument('--draco_skip_decode', default=True, type=str2bool, help='Draco teacherで復号を省略してbitだけ計測するか')
    parser.add_argument('--draco_force_point_cloud', default=True, type=str2bool, help='Draco -point_cloud を付けるか')
    parser.add_argument('--draco_merge_duplicated_points', default=True, type=str2bool, help='Draco teacher入力で重複点を統合する')
    parser.add_argument('--compression_surrogate_levels', default='4,6,8', type=str, help='Soft octree surrogate特徴に使う階層')
    parser.add_argument('--compression_surrogate_hidden_dim', default=128, type=int, help='圧縮サロゲートMLPの隠れ次元')
    parser.add_argument('--compression_surrogate_lr', default=3e-3, type=float, help='圧縮サロゲートのオンライン学習率')
    parser.add_argument('--compression_surrogate_weight_decay', default=1e-5, type=float, help='圧縮サロゲートのweight decay')
    parser.add_argument('--compression_surrogate_train_steps', default=2, type=int, help='教師更新時にサロゲートを教師bitに合わせて更新する回数')
    parser.add_argument('--compression_surrogate_warmup_steps', default=2, type=int, help='実ネットワーク更新前に行うSurrogate専用学習回数')
    parser.add_argument('--compression_surrogate_refresh_interval', default=1, type=int, help='何train stepごとに実圧縮教師を再計測するか(0なら初回以外は再計測しない)')
    parser.add_argument('--compression_surrogate_reuse_last_target', default=True, type=str2bool, help='同一train step内でのみ直近の実圧縮教師targetを再利用するか')
    parser.add_argument('--compression_surrogate_target_cache_entries', default=256, type=int, help='Surrogate教師targetのLRUキャッシュ数（生成側targetは同一train step内のみ有効）')
    parser.add_argument('--compression_surrogate_replay_steps', default=1, type=int, help='実圧縮を呼ばないstepでもreplay教師でSurrogateを更新する回数')
    parser.add_argument('--compression_surrogate_replay_batch', default=8, type=int, help='Surrogate replay学習のbatch数')
    parser.add_argument('--compression_surrogate_replay_entries', default=512, type=int, help='Surrogate replay bufferの最大件数')
    parser.add_argument('--compression_surrogate_forward_mode', default='teacher_ste', type=str, help='surrogate損失のforward値(surrogate/teacher_ste)')
    parser.add_argument(
        '--detach_surrogate_from_network',
        default=True,
        type=str2bool,
        help='TrueならSurrogate由来の勾配をNetwork側へ流さない。SurrogateはActual模倣・ログ・値比較専用にする',
    )
    parser.add_argument('--compression_surrogate_aux_node_weight', default=0.0, type=float, help='Surrogate debug用soft Octree node補助項の重み')
    parser.add_argument('--compression_surrogate_aux_single_weight', default=0.0, type=float, help='Surrogate debug用soft単一子ノード補助項の重み')
    parser.add_argument('--compression_surrogate_aux_in_objective', default=False, type=str2bool, help='soft Octree補助項を主圧縮objectiveへ混ぜるか。FalseならCompression Lossは実bit差分のみ')
    parser.add_argument('--compression_surrogate_log_soft_aux', default=True, type=str2bool, help='soft Octree node/single差分を主objectiveに混ぜず常時計算・ログする')
    parser.add_argument('--compression_soft_node_actuator_grad_weight', default=10.0, type=float, help='soft actuator rate proxyをnode項へSTEで戻す重み')
    parser.add_argument('--compression_soft_single_actuator_grad_weight', default=5.0, type=float, help='soft actuator rate proxyをsingle項へSTEで戻す重み')
    parser.add_argument('--compression_soft_bit_actuator_grad_weight', default=10.0, type=float, help='soft actuator rate proxyをbit proxy項へSTEで戻す重み')
    parser.add_argument('--compression_soft_prune_node_grad_weight', default=25.0, type=float, help='soft prune node proxyをnode項へSTEで戻す重み')
    parser.add_argument('--compression_soft_prune_single_grad_weight', default=20.0, type=float, help='soft prune single proxyをsingle項へSTEで戻す重み')
    parser.add_argument('--compression_soft_prune_bit_grad_weight', default=30.0, type=float, help='soft prune bit proxyをbit proxy項へSTEで戻す重み')
    parser.add_argument('--compression_soft_rate_point_weight', default=0.25, type=float, help='soft rate proxy内のpoint数差分重み')
    parser.add_argument('--compression_soft_rate_node_weight', default=0.10, type=float, help='soft rate proxy内のnode項重み')
    parser.add_argument('--compression_soft_rate_single_weight', default=0.05, type=float, help='soft rate proxy内のsingle項重み')
    parser.add_argument('--compression_soft_rate_sparsepcgc_weight', default=0.05, type=float, help='soft rate proxy内のSparsePCGC proxy重み')
    parser.add_argument('--compression_soft_rate_proxy_grad_weight', default=0.05, type=float, help='actual値forwardにsoft rate proxy勾配だけを足すSTE重み')
    parser.add_argument('--compression_soft_prune_rate_proxy_grad_weight', default=10.0, type=float, help='actual値forwardにsoft prune proxy勾配だけを足すSTE重み')
    parser.add_argument('--compression_octree_stat_depth', default=0, type=int, help='実圧縮debug用Octree統計の深さ(0なら点群から推定)')
    parser.add_argument('--compression_octree_stat_force', default=True, type=str2bool, help='圧縮器が返すnode/singleが0でも点群からOctree統計を補完する')
    parser.add_argument('--compression_surrogate_grad_clip', default=10.0, type=float, help='圧縮サロゲートの勾配クリップ')
    parser.add_argument('--compression_surrogate_empty_cache_after_update', default=True, type=str2bool, help='Surrogate更新直後にCUDA cacheを解放してGPU使用量の山を抑える')
    parser.add_argument('--compression_surrogate_empty_cache_threshold_mb', default=12288.0, type=float, help='CUDA cache解放を開始するreserved memory閾値(MB、0なら毎回)')
    parser.add_argument('--compression_surrogate_target_scale', default=100.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_pred_clip', default=0.0, type=float, help='サロゲートが予測する実圧縮bit差百分率のtanhクリップ（0で無効）')
    parser.add_argument('--surrogate_pred_clip_percent', default=-1.0, type=float, help='compression_surrogate_pred_clipの明示alias。負値ならcompression_surrogate_pred_clipを使う')
    parser.add_argument('--surrogate_target_clip_percent', default=0.0, type=float, help='実圧縮教師targetのclip幅。0で無効にしてraw percentを教師にする')
    parser.add_argument('--surrogate_use_log_bit_ratio_target', default=False, type=str2bool, help='Trueならraw percentではなくlog(after/before)をSurrogate教師に使う')
    parser.add_argument('--surrogate_log_bit_ratio_scale', default=100.0, type=float, help='log bit ratio教師を使う場合の倍率')
    parser.add_argument('--compression_surrogate_occ_gain', default=1.0, type=float, help='Soft occupancy変換のゲイン')
    parser.add_argument('--compression_surrogate_bit_weight', default=4.0, type=float, help='サロゲート教師損失のbit重み')
    parser.add_argument('--compression_surrogate_node_weight', default=1.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_single_weight', default=1.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_bpn_weight', default=1.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_entropy_weight', default=1.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_comp_bit_weight', default=1.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_comp_node_weight', default=0.25, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_comp_single_weight', default=0.25, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_comp_bpn_weight', default=0.25, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_comp_entropy_weight', default=0.25, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_loss_scale', default=100.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--sparsepcgc_aux_loss', default=True, type=str2bool, help='SparsePCGC向けactive coordinate/孤立proxy補助lossを使うか')
    parser.add_argument('--sparsepcgc_aux_backprop', default=False, type=str2bool, help='SparsePCGC soft proxy補助項を主圧縮lossへ勾配として流すか。Falseなら補助値は別ログに残し、実bit Surrogate本体を最適化する')
    parser.add_argument('--sparsepcgc_active_coord_weight', default=0.60, type=float, help='SparsePCGC補助loss内のactive coordinate削減項重み')
    parser.add_argument('--sparsepcgc_isolated_proxy_weight', default=0.25, type=float, help='SparsePCGC補助loss内の孤立voxel proxy項重み')
    parser.add_argument('--sparsepcgc_entropy_proxy_weight', default=0.15, type=float, help='SparsePCGC補助loss内のoccupancy entropy proxy項重み')
    parser.add_argument('--sparsepcgc_density_proxy_weight', default=0.05, type=float, help='SparsePCGC補助loss内のactive density proxy項重み')
    parser.add_argument('--sparsepcgc_aux_reward_clip', default=0.0, type=float, help='SparsePCGC補助lossのpercent項clip幅。0で無効')
    parser.add_argument('--sparsepcgc_corr_window', default=100, type=int, help='SparsePCGC proxy-actual相関を計算する直近サンプル数')
    parser.add_argument('--sparsepcgc_aux_gating', default=True, type=str2bool, help='proxy auxをbackpropする前にactual bitとのrolling一致度でgateする')
    parser.add_argument('--sparsepcgc_aux_min_corr', default=0.30, type=float, help='proxy aux backpropを許可する最小rolling相関')
    parser.add_argument('--sparsepcgc_aux_min_sign_match', default=0.50, type=float, help='proxy aux backpropを許可する最小rolling符号一致率')
    parser.add_argument('--sparsepcgc_aux_gating_window', default=100, type=int, help='proxy aux gateに使うrollingサンプル数')
    parser.add_argument('--sparsepcgc_disable_add', default=False, type=str2bool, help='SparsePCGCでは新規active coordinate増加を避けるため追加操作を既定で止める')
    parser.add_argument('--surrogate_step', default=0, type=int, help='main network更新前にSurrogateだけをactual teacherへfitさせるstep数')
    parser.add_argument('--surrogate_pretrain_lr', default=1e-4, type=float, help='Surrogate pretrain中のlearning rate')
    parser.add_argument('--surrogate_pretrain_actual_refresh_interval', default=10, type=int, help='Surrogate pretrain中のactual teacher refresh間隔')
    parser.add_argument('--surrogate_pretrain_freeze_network', default=True, type=str2bool, help='Surrogate pretrain中にmain networkをfreezeするか')
    parser.add_argument('--surrogate_pretrain_min_corr', default=-1.0, type=float, help='pretrain完了判定ログ用の最小actual/surrogate相関。負値で無効')
    parser.add_argument('--surrogate_pretrain_min_sign_match', default=-1.0, type=float, help='pretrain完了判定ログ用の最小sign match率。負値で無効')
    parser.add_argument('--surrogate_pretrain_min_fresh_samples', default=30, type=int, help='pretrain早期終了を判定する前に必要なfresh actual教師数')
    parser.add_argument('--surrogate_pretrain_min_abs_error', default=-1.0, type=float, help='pretrain早期終了用の平均abs error上限。負値で無効')
    parser.add_argument('--surrogate_pretrain_early_stop_patience', default=0, type=int, help='pretrain早期終了条件を連続で満たす必要step数。0で無効')
    parser.add_argument('--surrogate_pretrain_log_interval', default=10, type=int, help='Surrogate pretrainログ出力間隔')
    parser.add_argument('--surrogate_pretrain_print_interval', default=1, type=int, help='Surrogate pretrain進捗ログ出力間隔')
    parser.add_argument('--surrogate_pretrain_checkpoint', default=True, type=str2bool, help='Surrogate pretrain後にSurrogate stateを保存するか')
    parser.add_argument('--surrogate_pretrain_resume', default=True, type=str2bool, help='データセット別に保存済みのSurrogate事前学習stateを読み込むか')
    parser.add_argument('--surrogate_pretrain_skip_if_loaded', default=False, type=str2bool, help='保存済みSurrogateを読めた場合に事前学習を省略するか')
    parser.add_argument('--surrogate_pretrain_force_retrain', default=False, type=str2bool, help='保存済みSurrogateを無視して事前学習をやり直すか')
    parser.add_argument('--surrogate_pretrain_cache_dir', default=str((_LOG_ROOT / "surrogate_pretrain_cache").resolve()), type=str, help='データセット別Surrogate事前学習stateの保存先')
    parser.add_argument('--surrogate_registry_enabled', default=True, type=str2bool, help='指定形式の共有Surrogate重みを保存/読込するか')
    parser.add_argument('--surrogate_pretrained_root', default=str((_DATA_ROOT / "pretrained_surrogate").resolve()), type=str, help='共有Surrogate重みの保存root')
    parser.add_argument('--surrogate_data', default='', type=str, help='共有Surrogate重みファイル名に使うデータ名。空ならdataname_dataset_name')
    parser.add_argument('--surrogate_date', default=surrogate_date, type=str, help='読込対象Surrogate重みの日付。{surrogate_date}_{surrogate_time}.pthで探す')
    parser.add_argument('--surrogate_time', default=surrogate_time, type=str, help='読込対象Surrogate重みの時刻。{surrogate_date}_{surrogate_time}.pthで探す')
    parser.add_argument('--surrogate_registry_load_latest_if_missing', default=False, type=str2bool, help='指定Surrogateが無い場合に同一method内の最新Surrogateを読むか')
    parser.add_argument('--surrogate_pretrain_legacy_cache_fallback', default=False, type=str2bool, help='指定Surrogateが無い場合に旧fingerprint cacheを読むか')
    parser.add_argument('--surrogate_pretrain_use_replay', default=True, type=str2bool, help='pretrain中のnon-refresh stepでreplay教師更新を使うか')
    parser.add_argument('--surrogate_pretrain_replay_batch_size', default=16, type=int, help='pretrain replay更新のbatch size')
    parser.add_argument('--surrogate_pretrain_replay_steps', default=4, type=int, help='pretrain中の1stepあたりreplay更新回数')
    parser.add_argument('--surrogate_pretrain_replay_update_per_step', default=None, type=int, help='旧/明示alias: pretrain中の1stepあたりreplay更新回数')
    parser.add_argument('--surrogate_pretrain_replay_buffer_size', default=256, type=int, help='pretrain replay buffer最大件数')
    parser.add_argument('--surrogate_pretrain_replay_min_size', default=1, type=int, help='replay更新を開始する最小buffer件数')
    parser.add_argument('--surrogate_pretrain_allow_stale_target', default=True, type=str2bool, help='pretrain中にfresh teacherがないstepで直近targetをstale教師として使うか')
    parser.add_argument('--surrogate_pretrain_max_target_age', default=20, type=int, help='pretrain stale targetを使う最大step age')
    parser.add_argument('--surrogate_pretrain_skip_on_target_miss', default=False, type=str2bool, help='pretrain中にtarget/replayがないstepのsurrogate更新をskip扱いにするか')
    parser.add_argument('--surrogate_pretrain_sparsepcgc_debug_interval', default=10, type=int, help='pretrain中にSparsePCGC hard debug statsを取る間隔。0以下で無効')
    parser.add_argument('--surrogate_pretrain_mode', default='subtree', choices=['full', 'subtree', 'hybrid'], type=str, help='Surrogate pretrain方式(full/subtree/hybrid)')
    parser.add_argument('--surrogate_pretrain_subtree_stub_only', default=False, type=str2bool, help='旧互換用。Trueでもsubtree/hybrid実装が使える場合は実行する')
    parser.add_argument('--surrogate_pretrain_subtree_depth_min', default=-1, type=int, help='pretrain subtree深さの最小値。負値ならtrain_subtree設定を使う')
    parser.add_argument('--surrogate_pretrain_subtree_depth_max', default=-1, type=int, help='pretrain subtree深さの最大値。負値ならtrain_subtree設定を使う')
    parser.add_argument('--surrogate_pretrain_subtree_depth_percent_min', default=0.0, type=float, help='pretrain subtree深さを全体Octree深さ比で選ぶ時の最小割合')
    parser.add_argument('--surrogate_pretrain_subtree_depth_percent_max', default=0.50, type=float, help='pretrain subtree深さを全体Octree深さ比で選ぶ時の最大割合')
    parser.add_argument('--surrogate_pretrain_subtree_random_depth', default=True, type=str2bool, help='pretrain subtree深さをランダム/coverage samplingするか')
    parser.add_argument('--surrogate_pretrain_subtree_reuse_train_sampler', default=True, type=str2bool, help='通常trainのOctree subtree sampling helperをpretrainでも使うか')
    parser.add_argument('--surrogate_pretrain_skip_min_points_miss', default=False, type=str2bool, help='pretrain subtreeがtrain_subtree_min_pointsを満たせない時に1点subtreeへ落とさずskipするか')
    parser.add_argument('--surrogate_pretrain_subtree_steps_per_full', default=50, type=int, help='hybrid pretrainでfull校正1回あたりのsubtree step目安')
    parser.add_argument('--surrogate_pretrain_full_calibration_interval', default=50, type=int, help='hybrid pretrainでfull actual校正を行う間隔')
    parser.add_argument('--surrogate_pretrain_full_calibration_steps', default=1, type=int, help='hybrid pretrainの各calibration windowでfull actual校正に使うstep数')
    parser.add_argument('--surrogate_pretrain_use_full_teacher_for_subtree', default=False, type=str2bool, help='subtree pretrainでfull teacherを継承する実験flag。biasが大きいためdefault False')
    parser.add_argument('--surrogate_pretrain_subtree_teacher_type', default='local_actual', choices=['local_actual', 'local_proxy', 'inherited_full', 'none'], type=str, help='subtree/hybrid pretrainのteacher種別')
    parser.add_argument('--surrogate_pretrain_subtree_log_detail', default=True, type=str2bool, help='pretrain subtreeのdepth/点数/bboxをログするか')
    parser.add_argument('--surrogate_pretrain_store_local_proxy_replay', default=False, type=str2bool, help='local_proxy pretrain targetをactual mimic用replayへ保存するか。scale混同を避けるためdefault False')
    parser.add_argument('--surrogate_pretrain_max_wall_time_sec', default=0, type=float, help='pretrain最大実行秒数。0以下で無効')
    parser.add_argument('--surrogate_update_during_training', default=True, type=str2bool, help='通常学習中もSurrogate online更新を続けるか')
    parser.add_argument('--surrogate_update_interval', default=1, type=int, help='通常学習中のSurrogate optimizer更新間隔')
    parser.add_argument('--surrogate_joint_lr_scale', default=0.1, type=float, help='pretrain後の通常学習中Surrogate LR倍率')
    parser.add_argument('--surrogate_update_on_teacher_refresh_only', default=False, type=str2bool, help='Trueならteacher refresh時だけSurrogateを更新し、replay更新を止める')
    parser.add_argument('--surrogate_full_cloud_calib_interval', default=0, type=int, help='subtree学習中にfull-cloud actual teacher校正を入れる間隔。0で無効')
    parser.add_argument('--surrogate_full_cloud_calib_max_samples', default=1, type=int, help='full-cloud校正で使う最大サンプル数の予約設定')
    parser.add_argument('--surrogate_realign_on_low_corr', default=False, type=str2bool, help='低相関時のSurrogate再整列を有効化する実験flag')
    parser.add_argument('--surrogate_realign_min_corr', default=0.3, type=float, help='Surrogate再整列を検討する相関しきい値')
    parser.add_argument('--surrogate_realign_steps', default=0, type=int, help='低相関時に追加するSurrogate再整列step数。0ならログのみ')
    parser.add_argument('--surrogate_auto_freeze', default=False, type=str2bool, help='Surrogateが実bit教師に安定してfitしたらonline更新を一時停止する')
    parser.add_argument('--surrogate_freeze_abs_error', default=1.0, type=float, help='auto freezeに必要なbit percent abs error上限')
    parser.add_argument('--surrogate_freeze_train_loss', default=1.0, type=float, help='auto freezeに必要なSurrogate train loss上限')
    parser.add_argument('--surrogate_freeze_patience', default=8, type=int, help='auto freeze条件を連続で満たすfresh teacher回数')
    parser.add_argument('--surrogate_resume_abs_error', default=2.0, type=float, help='frozen Surrogateを再学習へ戻すbit percent abs error閾値')
    parser.add_argument('--surrogate_resume_train_loss', default=2.0, type=float, help='frozen Surrogateを再学習へ戻すtrain loss閾値')
    parser.add_argument('--compression_good_step_boost', default=False, type=str2bool, help='Surrogate安定後、実bitが改善したstepの主圧縮勾配を少し強める')
    parser.add_argument('--compression_good_step_boost_scale', default=1.5, type=float, help='実bit改善stepの主圧縮勾配倍率')
    parser.add_argument('--compression_good_step_prefreeze_scale', default=1.15, type=float, help='Surrogate freeze前でも十分fitした実bit改善stepだけに使う控えめな勾配倍率')
    parser.add_argument('--compression_good_step_prefreeze_max_train_loss', default=4.0, type=float, help='prefreeze改善step倍率を許可するSurrogate train loss上限')
    parser.add_argument('--compression_good_step_extra_surrogate_steps', default=4, type=int, help='実bit改善stepをSurrogateへ追加fitするstep数')
    parser.add_argument('--loss_grad_probe_enabled', default=False, type=str2bool, help='損失項ごとのmodule/操作別勾配CSVを出力するか')
    parser.add_argument('--loss_grad_probe_interval', default=1, type=int, help='損失項ごとの勾配CSVを何stepごとに出すか')
    parser.add_argument('--step_grad_log', default=True, type=str2bool, help='train.py内蔵の損失項ごとのmodule/操作別勾配CSVを出力するか')
    parser.add_argument('--step_grad_log_interval', default=50, type=int, help='train.py内蔵step_grad CSVを何stepごとに出すか')
    parser.add_argument('--step_grad_first_step_only', default=False, type=str2bool, help='step_grad CSVをglobal_step=0だけに制限するか')
    parser.add_argument('--compression_bad_step_penalty_scale', default=1.25, type=float, help='実bit悪化stepの主圧縮勾配倍率')
    parser.add_argument('--compression_boost_requires_surrogate_frozen', default=True, type=str2bool, help='good/bad勾配倍率をSurrogate frozen後だけ有効にする')
    parser.add_argument('--compression_boost_max_abs_error', default=1.0, type=float, help='good/bad勾配倍率を許可するSurrogate abs error上限')
    parser.add_argument('--sparsepcgc_enable_add_experiment', default=False, type=str2bool, help='SparsePCGCでもAddを実験的に許可する。defaultは必ずFalse')
    parser.add_argument('--sparsepcgc_add_only_when_compression_primary', default=True, type=str2bool, help='SparsePCGC Add実験をcompression_primary時だけ許可するか')
    parser.add_argument('--sparsepcgc_add_target_ratio', default=0.005, type=float, help='SparsePCGC Add実験のtarget add ratio')
    parser.add_argument('--sparsepcgc_add_max_ratio', default=0.50, type=float, help='SparsePCGC Add実験のmax add ratio')
    parser.add_argument('--sparsepcgc_add_warmup_steps', default=0, type=int, help='SparsePCGC Add実験のratio warmup step数')
    parser.add_argument('--sparsepcgc_add_use_candidate_score', default=True, type=str2bool, help='SparsePCGC Add実験で既存candidate scoreを使うか')
    parser.add_argument('--sparsepcgc_add_log_candidates', default=True, type=str2bool, help='SparsePCGC Add候補/scoreログを出すか')
    parser.add_argument('--sparsepcgc_add_active_coord_safety_gate', default=True, type=str2bool, help='SparsePCGC Add実験時にactive coord増加を安全gate/ログ対象にするか')
    parser.add_argument('--sparsepcgc_add_unique_coord_safety_gate', default=True, type=str2bool, help='SparsePCGC Add実験時にunique coord増加を安全gate/ログ対象にするか')
    parser.add_argument('--sparsepcgc_move_existing_target_only', default=False, type=str2bool, help='SparsePCGCでmove targetを既存occupied voxelへ寄せる旧実験設定。点潰れを避けるため既定ではFalse')
    parser.add_argument('--sparsepcgc_move_source_prior_weight', default=0.55, type=float, help='SparsePCGC時に原因診断scoreからAdjust source候補を起こす補助重み')
    parser.add_argument('--enable_voxel_collision_log', default=True, type=str2bool, help='SparsePCGC互換量子化後のVoxel衝突率ログを有効化する')
    parser.add_argument('--voxel_collision_log_interval', default=1, type=int, help='Voxel衝突率ログを記録するStep間隔')
    parser.add_argument('--voxel_collision_max_points', default=300000, type=int, help='Voxel衝突率ログで1点群あたり処理する最大点数。超過時は間引く')
    parser.add_argument('--voxel_collision_log_first_batch_only', default=True, type=str2bool, help='Voxel衝突率ログをBatch先頭だけに制限する')
    parser.add_argument('--voxel_collision_log_stages', default='input_gt,model_output_raw,compression_input', type=str, help='Voxel衝突率ログ対象stageのカンマ区切り一覧')
    parser.add_argument('--enable_sparsepcgc_empty_target_guard', default=True, type=str2bool, help='SparsePCGC互換量子化後に既存occupied targetへ入るAdjustをpreserveへ戻す')
    parser.add_argument('--enable_sparsepcgc_target_duplicate_guard', default=True, type=str2bool, help='SparsePCGC互換量子化後に同じtarget voxelへ集まるAdjustを1点だけ残してpreserveへ戻す')
    parser.add_argument(
        '--repair_move_relax_duplicate_guard_when_starved',
        default=True,
        type=str2bool,
        help='Adjust候補がduplicate guardで少なすぎる場合だけduplicate guardを緩める。empty target guardは維持する',
    )
    parser.add_argument('--sparsepcgc_empty_target_penalty_weight', default=0.0, type=float, help='既存occupied targetへ入るAdjustのsoft penalty重み')
    parser.add_argument('--sparsepcgc_target_duplicate_penalty_weight', default=0.0, type=float, help='同じtarget voxelへ複数Adjustするsoft penalty重み')
    parser.add_argument('--enable_sparsepcgc_occupancy_debug', default=False, type=str2bool, help='SparsePCGC本体の候補occupancy確率/label/bit推定ログをactual評価時だけ取得する')
    parser.add_argument('--sparsepcgc_occupancy_low_prob_threshold', default=0.1, type=float, help='SparsePCGC候補occupancyのtrue側確率を低確率とみなす閾値')
    parser.add_argument('--enable_sparsepcgc_exact_occupancy_teacher', default=False, type=str2bool, help='SparsePCGC本体定義のexact occupancy teacherログを有効化する')
    parser.add_argument('--sparsepcgc_exact_occupancy_interval', default=1, type=int, help='exact occupancy teacherを呼ぶStep間隔。0以下なら無効')
    parser.add_argument('--sparsepcgc_exact_teacher_mode', default='auto', type=str, help='exact teacherの基準(full_cloud/global_subtree/local_subtree/auto)')
    parser.add_argument('--enable_sparsepcgc_exact_occupancy_loss', default=False, type=str2bool, help='exact occupancy teacher由来の追加loss候補を有効化する')
    parser.add_argument('--sparsepcgc_exact_occupancy_loss_weight', default=0.0, type=float, help='exact occupancy NLL deltaのloss重み。既定0')
    parser.add_argument('--sparsepcgc_exact_bits_loss_weight', default=0.0, type=float, help='exact estimated bits deltaのloss重み。既定0')

    # proxyOctreeCompression
    parser.add_argument('--proxy_max_depth',     default=12,    type=int,   help='Octreeの最大深さ')
    parser.add_argument('--proxy_lambda_entropy', default=1,    type=float,   help='エントロピー項の重み')
    parser.add_argument('--proxy_lambda_node_count',   default=1,  type=float,   help='ノード数項の重み')
    parser.add_argument('--proxy_lambda_single_child', default=1,     type=float,   help='単一子ノード項の重み')
    parser.add_argument('--proxy_round_tau', default=0.12, type=float, help='soft丸めの温度パラメータ')
    parser.add_argument('--proxy_mass_to_occ_gain', default=1.0, type=float, help='質量→占有変換のスケール')
    parser.add_argument('--octattention_teacher_device', default='auto', type=str, help='OctAttention teacherの実行先(auto/cuda/cpu/balanced)')
    parser.add_argument('--compression_rate_metric', default='total_bits', type=str, help='圧縮率損失の基準(total_bits/bits_per_point/bits_per_input_point)')
    parser.add_argument('--disable_output_noise', default=True, type=str2bool, help='Trueなら出力点群へ圧縮損失用ノイズを加えずclean点群のまま使う')
    parser.add_argument('--use_uniform_noise', default=False, type=str2bool, help='train時のみ編集後・量子化前に加法的一様ノイズを入れるか')
    parser.add_argument('--noise_delta', default=1.0, type=float, help='一様ノイズ幅delta。u~Uniform(-delta/2,delta/2)。0以下ならcodec固有量子化幅へフォールバック')
    parser.add_argument('--log_step_time', default=True, type=str2bool, help='一定間隔でtrain/testの処理時間をログ出力するか')
    parser.add_argument('--log_gpu_memory', default=True, type=str2bool, help='一定間隔でGPUメモリ使用量をログ出力するか')
    parser.add_argument('--profile_interval', default=100, type=int, help='時間/GPU profileログの出力間隔')
    parser.add_argument('--train_subtree_anchor_on_min_points_miss', default=False, type=str2bool, help='subtreeがmin_pointsを満たせないdepthでは1点subtree学習を避けfull-cloud anchorへ切り替える')
    parser.add_argument('--for_better_log', default=True, type=str2bool, help='原因追跡用ForBetter.txtを通常ログとは別に出力するか')
    parser.add_argument('--for_better_log_interval', default=1, type=int, help='ForBetter.txtへtrain step診断を書く間隔')
    parser.add_argument('--for_better_spike_ratio', default=2.0, type=float, help='ForBetter.txtの急増event判定倍率')
    parser.add_argument('--for_better_spike_window', default=20, type=int, help='ForBetter.txtの急増event判定rolling window')
    parser.add_argument('--actual_eval_interval', default=1, type=int, help='train時のactual codec教師refresh間隔。0なら初回以外refreshしない')
    parser.add_argument('--disable_actual_codec_during_train', default=False, type=str2bool, help='train中のactual codec呼び出しをproxyへ置き換えて無効化するか')
    parser.add_argument('--actual_codec_fallback_to_proxy_on_error', default=True, type=str2bool, help='actual/surrogate teacherがtimeout等で失敗した場合にproxy lossへfallbackしてtrainを継続する')
    parser.add_argument('--skip_optimizer_on_actual_fallback', default=True, type=str2bool, help='actual/surrogate teacher失敗でproxy fallbackしたstepはoptimizer更新をスキップする')
    parser.add_argument('--actual_compression_guard', default=True, type=str2bool, help='episode平均のfresh actual圧縮損失が悪化し続けたらbestへ戻してLRを下げる')
    parser.add_argument('--actual_guard_patience', default=2, type=int, help='actual圧縮悪化を何episode連続で許容するか')
    parser.add_argument('--actual_guard_tolerance', default=0.25, type=float, help='best actual圧縮損失から何percentage pointの悪化まで許容するか')
    parser.add_argument('--actual_guard_decay_lr', default=False, type=str2bool, help='ActualCompressionGuard発火時にLRも下げるか。StepLRとの二重低下を避けるため既定False')
    parser.add_argument('--actual_guard_lr_decay', default=0.5, type=float, help='actual guard発動時のoptimizer LR倍率')
    parser.add_argument('--actual_guard_min_fresh', default=1, type=int, help='actual guardを判定する最低fresh actual計測数')
    parser.add_argument('--actual_guard_restore_best', default=True, type=str2bool, help='actual guard発動時にbest episode checkpointへ戻す')
    parser.add_argument('--actual_guard_improvement_epsilon', default=1e-6, type=float, help='actual guardのbest更新に必要な最小改善幅')
    parser.add_argument('--max_train_steps', default=0, type=int, help='デバッグ用: 0より大きい場合、そのglobal step数でtrain loopを早期終了する')
    parser.add_argument('--save_good_bad_cases', default=False, type=str2bool, help='actual deltaが大きく改善/悪化したstepのdebug summaryをCSV保存する')
    parser.add_argument('--save_proxy_actual_bad_cases', default=True, type=str2bool, help='proxy/cause scoreとactual bitの符号が逆のcaseをCSV保存する')
    parser.add_argument('--proxy_actual_bad_case_threshold', default=0.0, type=float, help='proxy-actual逆方向caseを保存する最小絶対値しきい値')
    parser.add_argument('--good_case_delta_threshold', default=-5.0, type=float, help='good caseとして保存するactual compression delta[%%]の閾値')
    parser.add_argument('--bad_case_delta_threshold', default=20.0, type=float, help='bad caseとして保存するactual compression delta[%%]の閾値')
    parser.add_argument('--max_saved_cases', default=64, type=int, help='good/bad case debug summaryの最大保存件数')
    parser.add_argument('--save_case_pointclouds', default=False, type=str2bool, help='予約: Trueならgood/bad caseの点群保存も許可する（現状はsummary CSVのみ）')
    parser.add_argument('--save_compression_metric_csv', default=True, type=str2bool, help='actual/surrogate/proxy圧縮metricを分離したstep CSVを保存する')
    parser.add_argument('--save_operation_metric_csv', default=True, type=str2bool, help='Add/Prune/Adjustのsoft/hard/effective統計step CSVを保存する')
    parser.add_argument('--operation_dead_grad_warn_threshold', default=1e-12, type=float, help='operation branch/amount勾配が死んだとみなすnormしきい値')
    parser.add_argument('--operation_dead_grad_warn_patience', default=20, type=int, help='operation勾配が低い状態が何step続いたらwarningを出すか')
    parser.add_argument('--repair_add_ratio_floor', default=0.0, type=float, help='Add操作が完全に死なないための弱いratio下限。0で無効')
    parser.add_argument('--save_checkpoint_metric_csv', default=True, type=str2bool, help='checkpoint判定に使うepisode metric CSVを保存する')
    parser.add_argument('--checkpoint_geom_gate', default=True, type=str2bool, help='actual-delta improved best保存時にgeometry gateを使う')
    parser.add_argument('--checkpoint_safety_gate', default=True, type=str2bool, help='actual-delta improved best保存時にrepair/node/single/operation safety gateを使う')
    parser.add_argument('--checkpoint_geom_rel_factor', default=1.5, type=float, help='stage内初回geometry lossに対するbest保存許容倍率')
    parser.add_argument('--checkpoint_geom_abs_max', default=0.0, type=float, help='best保存時のgeometry loss絶対上限。0以下で無効')
    parser.add_argument('--checkpoint_repair_rel_factor', default=0.0, type=float, help='stage内初回repair lossに対するbest保存許容倍率。0以下で無効')
    parser.add_argument('--checkpoint_repair_abs_max', default=10.0, type=float, help='best保存時のstructure repair loss絶対上限。0以下で無効')
    parser.add_argument('--checkpoint_node_abs_max', default=100.0, type=float, help='best保存時のnode loss絶対上限。0以下で無効')
    parser.add_argument('--checkpoint_single_abs_max', default=100.0, type=float, help='best保存時のsingle-child loss絶対上限。0以下で無効')
    parser.add_argument('--checkpoint_operation_ratio_max', default=100.0, type=float, help='best保存時のAdd/Prune/Adjust ratio[%%]上限。負値で無効')

    """Test"""
    parser.add_argument('--checkpoint', default=None, type=str, help='test.py用alias: 読み込む学習済みcheckpoint。指定時は--ckptを上書き')
    parser.add_argument('--data_dir', default=None, type=str, help='test.py用alias: 推論入力点群ディレクトリ。指定時は--input_dir_testを上書き')
    parser.add_argument('--output_log', default=None, type=str, help='test.pyの推論profile/点操作統計CSV保存先')
    parser.add_argument('--save_output_points', default=None, type=str2bool, help='test.py用alias: 編集後点群を保存するか。指定時は--save_test_plyを上書き')
    parser.add_argument('--output_dir', default=None, type=str, help='test.py用alias: 編集後点群保存先。指定時は--save_ply_dirを上書き')
    parser.add_argument('--max_test_samples', default=0, type=int, help='test.pyで処理する最大sample数。0以下なら全件')
    parser.add_argument('--input_dir_test', default=str(_data_subset_dir("ground")), type=str, help='テスト用入力点群のパス')
    parser.add_argument('--max_files_test', default=5, type=int, help='テスト時に読み込む最大ファイル数')
    parser.add_argument('--save_ply_dir', default=str(_data_subset_dir("test", dataname, dataset_name)), type=str, help='出力点群の保存先')
    parser.add_argument('--codec_eval_dir', default=str(_data_subset_dir("test")), type=str, help='encoder2decoder.py 系が参照する評価用PLYの同期先')
    parser.add_argument('--test_compute_loss', default=True, type=str2bool, help='test.pyで幾何・圧縮統計をログ出力するか')
    parser.add_argument('--skip_actual_codec', default=True, type=str2bool, help='test.pyではactual codec評価を省きproxy統計だけで高速評価するか')
    parser.add_argument('--codec_eval_interval', default=0, type=int, help='test.pyでactual codec評価を行う間隔。0なら無効、1なら全sample')
    parser.add_argument('--profile_test', default=False, type=str2bool, help='test.pyでsample別の処理時間/GPUメモリをログ出力するか')
    parser.add_argument('--save_test_ply', default=True, type=str2bool, help='test.pyで編集後PLYを保存するか')
    parser.add_argument('--test_drop_threshold', default=0.50, type=float, help='test.pyで点削除ゲートをhard化するしきい値。全点keep/全点dropになる場合はsum(final_w)ベースのexpected_keepへ自動フォールバック')
    parser.add_argument('--test_adjust_threshold', default=1e-6, type=float, help='test.pyで点が調整されたと数える最小移動距離')
    parser.add_argument('--test_inference_mode', default='full_cloud', type=str, help='推論方法(auto/full_cloud/subtree_merge/patch/direct/legacy)')
    parser.add_argument('--test_allow_subtree_merge', default=False, type=str2bool, help='Trueの時だけtest.pyでsubtree_merge推論を許可する')
    parser.add_argument('--test_auto_time_tolerance', default=0.10, type=float, help='auto選択で時間差がこの比率以内ならメモリ節約側を優先')
    parser.add_argument('--test_subtree_level', default=6, type=int, help='subtree_merge時に使うSubtree深さ(0ならtrain_subtree_level/repair_unit_level)')
    parser.add_argument('--test_subtree_batch_size', default=8, type=int, help='subtree_merge時に複数Subtreeをまとめて推論する数(Add有効時は安全のため1)')
    parser.add_argument('--test_subtree_min_points', default=4, type=int, help='subtree_merge時に各Subtreeへ最低限含めたい点数')
    parser.add_argument('--test_metric_max_points', default=8192, type=int, help='testログ用CD/D1/D2計算で使う最大点数（0で全点）')
    parser.add_argument('--test_metric_normal_k', default=16, type=int, help='testログ用D2PSNRの法線推定k近傍数')
    parser.add_argument('--test_compute_quality_metrics', default=True, type=str2bool, help='test.pyでCD/D1/D2品質指標を計算するか')

    """設定"""
    parser.add_argument('--seed', default=21, type=float, help='乱数シード')
    parser.add_argument('--deterministic', default=False, type=str2bool, help='再現性のためCUDAを固定するか')
    parser.add_argument('--num_points', default=8192, type=int, help='1パッチあたりの点数')
    parser.add_argument('--max_input_points', default=0, type=int, help='入力点数の上限（0で安全上限を使用）')
    parser.add_argument('--safe_max_input_points', default=0, type=int, help='max_input_points=0時にも適用する安全上限')
    parser.add_argument('--allow_unbounded_input', default=True, type=str2bool, help='Trueならmax_input_points=0で全点入力を許可する')
    parser.add_argument('--input_sampling', default='random', type=str, help='サンプリング方法')
    parser.add_argument('--split2patch', default=True, type=str2bool, help='点群をパッチ分割するか')
    parser.add_argument('--patch_rate', default=1.0, type=float, help='パッチ重なり率')
    parser.add_argument('--batch_size', default=1, type=int, help='バッチサイズ')
    parser.add_argument('--patch_batch_size', default=2, type=int, help='パッチ単位のバッチサイズ')
    parser.add_argument('--patch_parallel_mode', default='auto', type=str, help='パッチ並列化モード(auto/fixed/all)')
    parser.add_argument('--patch_parallel_points_budget_train', default=16384, type=int, help='train時に同時処理する総点数の目安')
    parser.add_argument('--patch_parallel_points_budget_test', default=16384, type=int, help='test時に同時処理する総点数の目安')
    parser.add_argument('--patch_cover_retry', default=4, type=int, help='カバーできない点を再試行する回数')
    parser.add_argument('--patch_build_mode', default='spatial_sort', type=str, help='パッチ構築方法(spatial_sort/fps_cover)')
    parser.add_argument('--patch_owned_ratio', default=0.9375, type=float, help='各パッチで固有に担当する点の割合')
    parser.add_argument('--patch_sort_grid_size', default=1024, type=int, help='spatial_sort時の粗い空間グリッド分解能')
    parser.add_argument('--patch_info_cache', default=True, type=str2bool, help='パッチ分割結果をCPUキャッシュして再利用するか')
    parser.add_argument('--train_patch_subset_enable', default=True, type=str2bool, help='train時にpatch subsetではなくOctree subtree subset学習を使うか')
    parser.add_argument('--train_subtree_level', default=4, type=int, help='train subtree深さ(0ならrepair_unit_levelを使う)')
    parser.add_argument('--train_subtree_randomize_level', default=True, type=str2bool, help='train時にsubtree深さを一定範囲でランダム化するか')
    parser.add_argument('--train_subtree_level_jitter', default=1, type=int, help='train_subtree_levelの前後に何段までランダム化を許すか')
    parser.add_argument('--train_subtree_level_min', default=4, type=int, help='train時subtree深さの最小値(0ならbase-jitter)')
    parser.add_argument('--train_subtree_level_max', default=0, type=int, help='train時subtree深さの最大値(0ならbase+jitter)')
    parser.add_argument('--train_subtree_random_full_range', default=False, type=str2bool, help='min/max未指定時にデータから推定した全Octree深さ範囲からランダムに選ぶ')
    parser.add_argument('--train_subtree_level_sampling', default='uniform_random', type=str, help='subtree深さサンプリング方法(uniform_random/coverage_cycle)')
    parser.add_argument('--train_subtree_level_curriculum', default=True, type=str2bool, help='学習中にsubtree深さ範囲を徐々に変える')
    parser.add_argument('--train_subtree_curriculum_fraction', default=1.0, type=float, help='全学習stepのうちsubtree深さcurriculumに使う割合（1.0なら全stepで徐々に変化）')
    parser.add_argument('--train_subtree_curriculum_direction', default='deep_to_shallow', type=str, help='subtree深さcurriculumの向き(deep_to_shallow/shallow_to_deep)')
    parser.add_argument('--train_subtree_depth_percent_curriculum', default=True, type=str2bool, help='全体Octree深さに対する割合でsubtree深さ範囲を決める')
    parser.add_argument('--train_subtree_depth_percent_start', default='0.0,0.50', type=str, help='学習開始時のsubtree深さ割合範囲(min,max)')
    parser.add_argument('--train_subtree_depth_percent_end', default='0.0,0.50', type=str, help='学習終了時のsubtree深さ割合範囲(min,max)')
    parser.add_argument('--train_subtree_min_points', default=5, type=int, help='train時に優先的に選ぶsubtreeの最小点数（満たす候補が無ければフォールバック）')
    parser.add_argument('--train_patch_subset_patches_per_step', default=1, type=int, help='1 stepで処理するsubtree数')
    parser.add_argument('--train_patch_subset_anchor_interval', default=32, type=int, help='subtree subset学習時に何stepごとにfull-cloud anchor学習を挟むか(0なら間隔指定なし)')
    parser.add_argument('--train_subtree_full_cloud_prob', default=0.03, type=float, help='subtree subset学習時に確率的にfull-cloud anchorへ切り替える確率')
    parser.add_argument('--train_patch_subset_sampling', default='coverage_cycle', type=str, help='subtree subset学習の選択方法(coverage_cycle)')
    parser.add_argument('--train_patch_subset_log', default=True, type=str2bool, help='subtree subset学習の選択状況をログ出力するか')
    parser.add_argument('--train_subtree_stat_log_limit', default=16, type=int, help='SubtreeSelectionログでOctree統計を計算する最大subtree数')
    parser.add_argument('--num_workers', default=4, type=int, help='データローダのワーカー数')
    parser.add_argument('--pin_memory', default=True, type=str2bool, help='CPU→GPU転送高速化のためメモリ固定するか')
    parser.add_argument('--persistent_workers', default=True, type=str2bool, help='ワーカーを維持するか')
    parser.add_argument('--dataset_cache', default=False, type=str2bool, help='データセットをメモリにキャッシュするか')
    parser.add_argument('--ply_loader', default='numpy', type=str, help='PLY読み込み方法(numpy/open3d/auto)')
    parser.add_argument('--mp_start_method', default='auto', type=str, help='マルチプロセス起動方法')
    parser.add_argument('--weight_decay', default=0, type=float, help='重み減衰')
    parser.add_argument('--bptt', default=1024, type=float, help='（内部用パラメータ）')
    parser.add_argument('--parallel', default=False, type=str2bool, help='並列モデルを使うか')
    parser.add_argument('--module_bn_use_running_stats', default=False, type=str2bool, help='Encoder以外のBatchNormでrunning statsを使うか')
    parser.add_argument('--use_tf32', default=True, type=str2bool, help='TF32を使用するか')
    parser.add_argument('--use_amp', default=True, type=str2bool, help='混合精度学習を使うか')
    parser.add_argument('--amp_dtype', default='auto', type=str, help='AMPのデータ型')
    parser.add_argument('--amp_init_scale', default=1.0, type=float, help='GradScaler初期値')
    parser.add_argument('--amp_overflow_patience', default=2, type=int, help='オーバーフロー許容回数')
    parser.add_argument('--cache_frozen_inputs', default=True, type=str2bool, help='Encoder出力をキャッシュするか')
    parser.add_argument('--cache_gt_loss', default=True, type=str2bool, help='GT側損失をキャッシュするか')
    parser.add_argument('--cache_max_entries', default=192, type=int, help='キャッシュ最大数')
    parser.add_argument('--cache_max_memory_mb', default=8192, type=int, help='キャッシュ最大メモリ（MB）')
    parser.add_argument('--auto_disable_partial_frozen_cache', default=True, type=str2bool, help='キャッシュ不足時に自動無効化するか')
    parser.add_argument('--clear_main_ply_cache_for_workers', default=True, type=str2bool, help='メモリ重複を防ぐためキャッシュ削除')
    parser.add_argument('--warmup_frozen_cache', default=False, type=str2bool, help='事前にキャッシュを作るか')
    parser.add_argument('--warmup_gt_cache', default=False, type=str2bool, help='GTキャッシュを事前生成するか')
    parser.add_argument('--warmup_max_files', default=0, type=int, help='ウォームアップ対象ファイル数')
    parser.add_argument('--warmup_max_seconds', default=0, type=float, help='ウォームアップ最大時間')
    parser.add_argument('--warmup_log_rate', default=8, type=int, help='ウォームアップログ間隔')
    parser.add_argument('--log_flush_every', default=32, type=int, help='ログ書き込みフラッシュ間隔')
    parser.add_argument('--log_sync_every', default=0, type=int, help='ログ同期間隔')
    parser.add_argument('--verbose_step_logs', default=True, type=str2bool, help='詳細ログを出すか')
    parser.add_argument('--epoch_plot_rate', default=1, type=int, help='エポックごとのプロット保存間隔')
    parser.add_argument('--episode_plot_rate', default=1, type=int, help='エピソードごとのプロット保存間隔')
    parser.add_argument('--plot_max_points', default=512, type=int, help='1枚のグラフに描画する最大点数（超過時は等間隔に間引く）')
    parser.add_argument('--plot_skip_outlier_steps', default=True, type=str2bool, help='Trueなら極端に大きいstep値をプロット履歴から除外する')
    parser.add_argument('--plot_outlier_abs_threshold', default=1e6, type=float, help='この絶対値を超えるstep値をプロットから除外する閾値（0以下で無効）')
    parser.add_argument('--plot_outlier_rel_factor', default=100.0, type=float, help='直近履歴の中央値に対する倍率閾値（0以下で無効）')
    parser.add_argument('--plot_outlier_min_history', default=4, type=int, help='相対外れ値判定を始めるまでに必要な履歴数')
    parser.add_argument('--plot_outlier_history_window', default=64, type=int, help='相対外れ値判定で参照する直近履歴数（0で全履歴）')
    parser.add_argument('--plot_outlier_min_scale', default=1.0, type=float, help='相対外れ値判定で使う中央値の下限値')
    parser.add_argument('--retain_debug_tensors', default=False, type=str2bool, help='中間勾配を保持するか')
    parser.add_argument('--debug_grad_flow', default=False, type=str2bool, help='勾配ノルムをログ出力するか')
    parser.add_argument('--debug_grad_flow_rate', default=1, type=int, help='勾配ログの出力間隔')
    parser.add_argument('--train_grad_clip', default=0.0, type=float, help='学習時の勾配クリップ値（0で無効）')
    parser.add_argument('--skip_optimizer_on_nonfinite_grad', default=True, type=str2bool, help='Lossが有限でも勾配にNaN/Infがあるstepはoptimizer更新をスキップする')
    parser.add_argument('--nonfinite_grad_log_param_limit', default=8, type=int, help='非有限勾配を検出したときにログへ出すパラメータ名の最大数')
    parser.add_argument('--debug_timing', default=False, type=str2bool, help='ステップ内の時間内訳をログ出力するか')
    parser.add_argument('--sparsepcgc_hard_debug_interval', default=0, type=int, help='SparsePCGCの重いhard統計を何Stepごとに収集するか(0で無効)')
    parser.add_argument('--sparsepcgc_hard_debug_on_log', default=False, type=str2bool, help='Trueなら通常Stepログ時にもSparsePCGC hard統計を収集する')
    parser.add_argument('--mail_notify', default=False, type=str2bool, help='学習イベントをメール通知するか')
    parser.add_argument('--mail_to', default='maeshu0619@gmail.com', type=str, help='通知メールの宛先')
    parser.add_argument('--mail_from', default='maeshu0619@gmail.com', type=str, help='通知メールの送信元')
    parser.add_argument('--mail_smtp_host', default='smtp.gmail.com', type=str, help='SMTPホスト。空ならsendmailを試す')
    parser.add_argument('--mail_smtp_port', default=587, type=int, help='SMTPポート')
    parser.add_argument('--mail_smtp_user', default='', type=str, help='SMTPユーザ')
    parser.add_argument('--mail_smtp_password_env', default='MYNET_MAIL_PASSWORD', type=str, help='SMTPパスワードを読む環境変数名')
    parser.add_argument('--mail_use_tls', default=True, type=str2bool, help='SMTP STARTTLSを使うか')
    parser.add_argument('--mail_timeout', default=10.0, type=float, help='メール送信タイムアウト秒')
    parser.add_argument('--mail_sendmail_path', default='/usr/sbin/sendmail', type=str, help='sendmailコマンドのパス')

    args = parser.parse_args()
    # more_training=True のときに読む追加学習用checkpointパスを確定する
    # グローバル設定値をログで確認しやすいようにargsへ保存する
    args.method_com = method_com
    args.model_name = model_name
    if bool(getattr(args, "more_training", False)):
        if not str(getattr(args, "more_training_ckpt", "")).strip():
            args.more_training_ckpt = _more_training_checkpoint_path(
                args.pretrained_date,
                args.pretrained_time,
                method_value=method_com,
                model_stem=model_name,
            )
        else:
            args.more_training_ckpt = str(Path(os.path.expanduser(args.more_training_ckpt)).resolve())
    else:
        args.more_training_ckpt = ""

    args.repair_operation_amount_logit_weight = max(float(getattr(args, "repair_operation_amount_logit_weight", 0.05)), 0.0)
    args.repair_operation_amount_logit_scale = max(float(getattr(args, "repair_operation_amount_logit_scale", 6.0)), 1e-6)
    args.repair_operation_amount_target_prob_max = min(
        max(float(getattr(args, "repair_operation_amount_target_prob_max", 0.98)), 0.50),
        1.0 - 1e-4,
    )
    args.w_com = max(float(getattr(args, "w_com", 10.0)), 0.0)
    args.repair_drop_where_actuator_weight = max(float(getattr(args, "repair_drop_where_actuator_weight", 0.1)), 0.0)
    args.repair_add_where_actuator_weight = max(float(getattr(args, "repair_add_where_actuator_weight", 0.3)), 0.0)
    args.repair_move_where_actuator_weight = max(float(getattr(args, "repair_move_where_actuator_weight", 0.3)), 0.0)
    args.repair_where_downstream_grad_scale = max(float(getattr(args, "repair_where_downstream_grad_scale", 1.0)), 1.0)
    args.repair_drop_where_downstream_grad_scale = max(float(getattr(args, "repair_drop_where_downstream_grad_scale", args.repair_where_downstream_grad_scale)), 1.0)
    args.repair_add_where_downstream_grad_scale = max(float(getattr(args, "repair_add_where_downstream_grad_scale", args.repair_where_downstream_grad_scale)), 1.0)
    args.repair_move_where_downstream_grad_scale = max(float(getattr(args, "repair_move_where_downstream_grad_scale", args.repair_where_downstream_grad_scale)), 1.0)

    args._add_cli_provided = _cli_option_was_provided("--add")
    args._use_amp_cli_provided = _cli_option_was_provided("--use_amp")
    if not _cli_option_was_provided("--input_dir_test"):
        args.input_dir_test = str(_data_subset_dir("ground", args.dataname, args.dataset_name))
    if not _cli_option_was_provided("--save_ply_dir"):
        args.save_ply_dir = str(_data_subset_dir("test", args.dataname, args.dataset_name).resolve())
    if not _cli_option_was_provided("--codec_eval_dir"):
        args.codec_eval_dir = str(_data_subset_dir("test", args.dataname, args.dataset_name))
    if getattr(args, "checkpoint", None):
        args.ckpt = args.checkpoint
    if getattr(args, "data_dir", None):
        args.input_dir_test = args.data_dir
    if getattr(args, "output_dir", None):
        args.save_ply_dir = args.output_dir
    if getattr(args, "save_output_points", None) is not None:
        args.save_test_ply = bool(args.save_output_points)
    if getattr(args, "output_log", None) is None:
        args.output_log = str((_LOG_ROOT / args.date / "MyNetwork_test" / "profile" / f"{args.time}.csv").resolve())

    args.octree_ctx_dim = max(int(args.octree_ctx_dim), 1)
    if _cli_option_was_provided("--w_prun") and not _cli_option_was_provided("--w_attr"):
        args.w_attr = float(args.w_prun)
    if _cli_option_was_provided("--w_add") and not _cli_option_was_provided("--w_policy"):
        args.w_policy = float(args.w_add)
    if _cli_option_was_provided("--w_dis") and not _cli_option_was_provided("--w_actuator"):
        args.w_actuator = float(args.w_dis)
    args.w_attr = float(args.w_attr)
    args.w_policy = float(args.w_policy)
    args.w_actuator = float(args.w_actuator)
    args.w_prun = args.w_attr
    args.w_add = args.w_policy
    args.w_dis = args.w_actuator
    args.w_repair = args.w_actuator

    args.loss_mode = str(getattr(args, "loss_mode", "legacy_total")).strip().lower()
    if args.loss_mode not in {"legacy_total", "compression_primary"}:
        raise ValueError("--loss_mode must be legacy_total or compression_primary")
    args.compression_primary_warmup_steps = max(
        int(getattr(args, "compression_primary_warmup_steps", 0)),
        0,
    )
    for _name in (
        "cp_lambda_geom",
        "cp_lambda_single",
        "cp_lambda_nodes",
        "cp_lambda_actuator",
        "cp_lambda_sparsepcgc",
        "cp_lambda_op",
    ):
        setattr(args, _name, max(float(getattr(args, _name, 0.0)), 0.0))
    for _name in (
        "cp_tau_geom",
        "cp_tau_single",
        "cp_tau_nodes",
        "cp_tau_actuator",
        "cp_tau_sparsepcgc",
    ):
        setattr(args, _name, float(getattr(args, _name, 0.0)))
    args.cp_use_stage_factors = bool(getattr(args, "cp_use_stage_factors", False))
    args.cp_force_joint_actuator = bool(getattr(args, "cp_force_joint_actuator", True))
    args.cp_log_grad_terms = bool(getattr(args, "cp_log_grad_terms", True))

    stage = str(args.training_stage).strip().lower()
    if stage not in {"diagnosis", "joint"}:
        raise ValueError(f"--training_stage must be diagnosis or joint (got {args.training_stage})")
    args.training_stage = stage
    args.diagnosis_episode_ratio = min(max(float(args.diagnosis_episode_ratio), 0.0), 1.0)
    args.diagnosis_episodes = max(int(args.diagnosis_episodes), 0)

    mode_alias = {
        "hard_ste": "ste_hard",
        "soft": "weighted_soft",
        "legacy": "weighted_soft",
    }
    discrete_loss_mode = str(args.discrete_loss_mode).strip().lower()
    args.discrete_loss_mode = mode_alias.get(discrete_loss_mode, discrete_loss_mode)
    if args.discrete_loss_mode not in {"ste_hard", "hard", "weighted_soft"}:
        raise ValueError(
            "--discrete_loss_mode must be one of: ste_hard, hard, weighted_soft "
            f"(got {args.discrete_loss_mode})"
        )

    args.compression_rate_metric = str(args.compression_rate_metric).strip().lower()
    if args.compression_rate_metric not in {"total_bits", "bits_per_point", "bits_per_input_point"}:
        raise ValueError(
            "--compression_rate_metric must be one of: total_bits, bits_per_point, "
            f"bits_per_input_point (got {args.compression_rate_metric})"
        )

    args.compress = str(args.compress).strip()
    compress_key = _compress_key(args.compress)
    if compress_key == "sparsepcgc":
        args.compress = "SparsePCGC"
    elif compress_key == "gpcc":
        args.compress = "G-PCC"
    elif compress_key == "octattention":
        args.compress = "OctAttention"

    args.compression_loss_backend = str(args.compression_loss_backend).strip().lower()
    if bool(args.surrogate):
        if compress_key == "sparsepcgc":
            args.compression_loss_backend = "sparsepcgc_surrogate"
        elif compress_key == "gpcc":
            args.compression_loss_backend = "gpcc_surrogate"
        elif compress_key == "draco":
            args.compression_loss_backend = "draco_surrogate"
        elif compress_key == "octattention":
            args.compression_loss_backend = "octattention_surrogate"
        else:
            raise ValueError(
                "--surrogate=True currently supports --compress OctAttention, SparsePCGC, or G-PCC "
                f"(got {args.compress})"
            )
    args.method_name = str(getattr(args, "method_name", method_name)).strip() or method_name
    args.surrogate_name = _compress_display_name(args.compress)
    args.run_name = str(getattr(args, "run_name", "")).strip()
    if not args.run_name:
        args.run_name = f"{args.time}_{args.surrogate_name}"
    args.pretrained_date = str(getattr(args, "pretrained_date", pretrained_date)).strip()
    args.pretrained_time = str(getattr(args, "pretrained_time", pretrained_time)).strip()
    if not _cli_option_was_provided("--ckpt") and not getattr(args, "checkpoint", None):
        args.ckpt = _pretrained_checkpoint_path(
            args.pretrained_date,
            args.pretrained_time,
            args.compress,
            model_stem=model_name,
        )
    if not _cli_option_was_provided("--out_path"):
        args.out_path = str((_LOG_ROOT / args.date / "MyNetwork_train" / "pretrained" / args.run_name).resolve())
    actual_bit_backends = {
        "octattention_actual",
        "octattention_actual_ste",
        "octattention_surrogate",
        "sparsepcgc_actual",
        "sparsepcgc_actual_ste",
        "sparsepcgc_surrogate",
        "gpcc_actual",
        "gpcc_actual_ste",
        "gpcc_surrogate",
        "draco_actual",
        "draco_actual_ste",
        "draco_surrogate",
    }
    if args.compression_loss_backend in actual_bit_backends:
        args.compression_rate_metric = "total_bits"
    args.plot_max_points = max(int(args.plot_max_points), 2)
    args.plot_skip_outlier_steps = bool(getattr(args, "plot_skip_outlier_steps", True))
    args.plot_outlier_abs_threshold = float(getattr(args, "plot_outlier_abs_threshold", 0.0))
    args.plot_outlier_rel_factor = float(getattr(args, "plot_outlier_rel_factor", 0.0))
    args.plot_outlier_min_history = max(int(getattr(args, "plot_outlier_min_history", 0)), 0)
    args.plot_outlier_history_window = max(int(getattr(args, "plot_outlier_history_window", 0)), 0)
    args.plot_outlier_min_scale = max(float(getattr(args, "plot_outlier_min_scale", 0.0)), 0.0)
    args.save_compression_metric_csv = bool(getattr(args, "save_compression_metric_csv", True))
    args.save_operation_metric_csv = bool(getattr(args, "save_operation_metric_csv", True))
    args.save_checkpoint_metric_csv = bool(getattr(args, "save_checkpoint_metric_csv", True))
    args.checkpoint_geom_gate = bool(getattr(args, "checkpoint_geom_gate", True))
    args.checkpoint_safety_gate = bool(getattr(args, "checkpoint_safety_gate", True))
    args.checkpoint_geom_rel_factor = max(float(getattr(args, "checkpoint_geom_rel_factor", 1.5)), 0.0)
    args.checkpoint_geom_abs_max = max(float(getattr(args, "checkpoint_geom_abs_max", 0.0)), 0.0)
    args.checkpoint_repair_rel_factor = max(float(getattr(args, "checkpoint_repair_rel_factor", 0.0)), 0.0)
    args.checkpoint_repair_abs_max = max(float(getattr(args, "checkpoint_repair_abs_max", 10.0)), 0.0)
    args.checkpoint_node_abs_max = max(float(getattr(args, "checkpoint_node_abs_max", 100.0)), 0.0)
    args.checkpoint_single_abs_max = max(float(getattr(args, "checkpoint_single_abs_max", 100.0)), 0.0)
    args.checkpoint_operation_ratio_max = float(getattr(args, "checkpoint_operation_ratio_max", 100.0))
    args.operation_count_drop_threshold = min(
        max(float(getattr(args, "operation_count_drop_threshold", 0.5)), 0.0),
        1.0,
    )
    args.operation_count_adjust_threshold = max(
        float(getattr(args, "operation_count_adjust_threshold", 1e-6)),
        0.0,
    )
    valid_backends = {
        "proxy",
        "octattention_actual",
        "octattention_actual_ste",
        "octattention_surrogate",
        "sparsepcgc_actual",
        "sparsepcgc_actual_ste",
        "sparsepcgc_surrogate",
        "gpcc_actual",
        "gpcc_actual_ste",
        "gpcc_surrogate",
        "draco_actual",
        "draco_actual_ste",
        "draco_surrogate",
    }
    if args.compression_loss_backend not in valid_backends:
        raise ValueError(
            "--compression_loss_backend must be one of: proxy, octattention_actual, "
            "octattention_actual_ste, octattention_surrogate, sparsepcgc_actual, "
            "sparsepcgc_actual_ste, sparsepcgc_surrogate, gpcc_actual, "
            "gpcc_actual_ste, gpcc_surrogate, draco_actual, draco_actual_ste, "
            f"draco_surrogate (got {args.compression_loss_backend})"
        )

    teacher_device = str(args.octattention_teacher_device).strip().lower()
    valid_teacher_devices = {"auto", "cpu", "balanced", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    if teacher_device not in valid_teacher_devices:
        raise ValueError(
            "--octattention_teacher_device must be one of: auto, cpu, balanced, cuda, cuda:0..cuda:3 "
            f"(got {args.octattention_teacher_device})"
        )
    if teacher_device == "auto" and args.compression_loss_backend.startswith("octattention_"):
        teacher_device = "balanced"
    args.octattention_teacher_device = teacher_device

    args.sparsepcgc_mode = str(getattr(args, "sparsepcgc_mode", "dense_lossless")).strip().lower()
    valid_sparsepcgc_modes = {"dense_lossless", "dense_lossy", "sparse_lossless", "sparse_lossy_gpcc"}
    if args.sparsepcgc_mode not in valid_sparsepcgc_modes:
        raise ValueError(
            "--sparsepcgc_mode must be one of: dense_lossless, dense_lossy, "
            f"sparse_lossless, sparse_lossy_gpcc (got {args.sparsepcgc_mode})"
        )
    args.sparsepcgc_device = str(getattr(args, "sparsepcgc_device", "auto")).strip().lower()
    args.sparsepcgc_timeout = max(float(getattr(args, "sparsepcgc_timeout", 600.0)), 1.0)
    args.sparsepcgc_match_qs = bool(getattr(args, "sparsepcgc_match_qs", True))
    if _compress_key(getattr(args, "compress", "")) == "sparsepcgc" and args.sparsepcgc_match_qs:
        voxel_cli = _cli_option_was_provided("--sparsepcgc_voxel_size")
        quant_cli = _cli_option_was_provided("--sparsepcgc_pos_quantscale")
        if not voxel_cli and not quant_cli:
            args.sparsepcgc_voxel_size = float(getattr(args, "qs", 1.0))
            args.sparsepcgc_pos_quantscale = 1
        elif not voxel_cli and quant_cli:
            args.sparsepcgc_voxel_size = float(getattr(args, "qs", 1.0)) / max(float(getattr(args, "sparsepcgc_pos_quantscale", 1)), 1.0)
    args.sparsepcgc_voxel_size = max(float(getattr(args, "sparsepcgc_voxel_size", 1.0)), 1e-12)
    args.sparsepcgc_pos_quantscale = max(int(getattr(args, "sparsepcgc_pos_quantscale", 1)), 1)
    args.sparsepcgc_effective_qs = float(args.sparsepcgc_voxel_size) * float(args.sparsepcgc_pos_quantscale)
    args.sparsepcgc_dense_scale_ae_list = _parse_csv_ints(args.sparsepcgc_dense_scale_ae_list)
    args.sparsepcgc_dense_scale_sr_list = _parse_csv_ints(args.sparsepcgc_dense_scale_sr_list)
    args.sparsepcgc_pos_quantscale_list = _parse_csv_ints(args.sparsepcgc_pos_quantscale_list)
    args.gpcc_timeout = max(float(getattr(args, "gpcc_timeout", 120.0)), 1.0)
    args.gpcc_match_qs = bool(getattr(args, "gpcc_match_qs", True))
    args.gpcc_prequantize = bool(getattr(args, "gpcc_prequantize", True))
    if _compress_key(getattr(args, "compress", "")) == "gpcc" and args.gpcc_match_qs:
        if not _cli_option_was_provided("--gpcc_effective_qs"):
            args.gpcc_effective_qs = float(getattr(args, "qs", 1.0))
    args.gpcc_effective_qs = max(float(getattr(args, "gpcc_effective_qs", 1.0)), 1e-12)
    args.gpcc_disable_attribute_coding = bool(getattr(args, "gpcc_disable_attribute_coding", True))
    args.gpcc_merge_duplicated_points = bool(getattr(args, "gpcc_merge_duplicated_points", True))
    args.draco_timeout = max(float(getattr(args, "draco_timeout", 120.0)), 1.0)
    args.draco_match_qs = bool(getattr(args, "draco_match_qs", True))
    args.draco_prequantize = bool(getattr(args, "draco_prequantize", True))
    if _compress_key(getattr(args, "compress", "")) == "draco" and args.draco_match_qs:
        if not _cli_option_was_provided("--draco_effective_qs"):
            args.draco_effective_qs = float(getattr(args, "qs", 1.0))
    args.draco_effective_qs = max(float(getattr(args, "draco_effective_qs", 1.0)), 1e-12)
    if (
        _compress_key(getattr(args, "compress", "")) == "draco"
        and args.draco_prequantize
        and not _cli_option_was_provided("--draco_position_quantization_bits")
    ):
        args.draco_position_quantization_bits = 0
    args.draco_position_quantization_bits = int(getattr(args, "draco_position_quantization_bits", 0))
    if args.draco_position_quantization_bits < 0:
        raise ValueError("--draco_position_quantization_bits must be >= 0")
    args.draco_compression_level = min(max(int(getattr(args, "draco_compression_level", 7)), 0), 10)
    args.draco_skip_decode = bool(getattr(args, "draco_skip_decode", True))
    args.draco_force_point_cloud = bool(getattr(args, "draco_force_point_cloud", True))
    args.draco_merge_duplicated_points = bool(getattr(args, "draco_merge_duplicated_points", True))
    args.compression_surrogate_train_steps = max(int(getattr(args, "compression_surrogate_train_steps", 0)), 0)
    args.compression_surrogate_warmup_steps = max(
        int(getattr(args, "compression_surrogate_warmup_steps", args.compression_surrogate_train_steps)),
        0,
    )
    args.compression_surrogate_refresh_interval = max(
        int(getattr(args, "compression_surrogate_refresh_interval", 0)),
        0,
    )
    args.compression_surrogate_target_cache_entries = max(
        int(getattr(args, "compression_surrogate_target_cache_entries", 0)),
        0,
    )
    args.compression_surrogate_reuse_last_target = bool(
        getattr(args, "compression_surrogate_reuse_last_target", True)
    )
    args.compression_surrogate_replay_steps = max(
        int(getattr(args, "compression_surrogate_replay_steps", 0)),
        0,
    )
    args.compression_surrogate_replay_batch = max(
        int(getattr(args, "compression_surrogate_replay_batch", 1)),
        1,
    )
    args.compression_surrogate_replay_entries = max(
        int(getattr(args, "compression_surrogate_replay_entries", 0)),
        0,
    )
    args.compression_surrogate_empty_cache_after_update = bool(getattr(args, "compression_surrogate_empty_cache_after_update", True)) # Surrogate更新後のCUDA cache解放を有効化する
    args.compression_surrogate_empty_cache_threshold_mb = max(float(getattr(args, "compression_surrogate_empty_cache_threshold_mb", 12288.0)), 0.0) # CUDA cache解放を始めるreserved memory閾値を正規化する
    args.surrogate_step = max(int(getattr(args, "surrogate_step", 0)), 0)
    args.surrogate_pretrain_lr = max(float(getattr(args, "surrogate_pretrain_lr", 1e-4)), 0.0)
    args.surrogate_pretrain_actual_refresh_interval = max(
        int(getattr(args, "surrogate_pretrain_actual_refresh_interval", 10)),
        0,
    )
    args.surrogate_pretrain_freeze_network = bool(getattr(args, "surrogate_pretrain_freeze_network", True))
    args.surrogate_pretrain_min_corr = float(getattr(args, "surrogate_pretrain_min_corr", -1.0))
    args.surrogate_pretrain_min_sign_match = float(getattr(args, "surrogate_pretrain_min_sign_match", -1.0))
    args.surrogate_pretrain_min_fresh_samples = max(
        int(getattr(args, "surrogate_pretrain_min_fresh_samples", 30)),
        0,
    )
    args.surrogate_pretrain_min_abs_error = float(getattr(args, "surrogate_pretrain_min_abs_error", -1.0))
    args.surrogate_pretrain_early_stop_patience = max(
        int(getattr(args, "surrogate_pretrain_early_stop_patience", 0)),
        0,
    )
    args.surrogate_pretrain_log_interval = max(int(getattr(args, "surrogate_pretrain_log_interval", 10)), 1)
    args.surrogate_pretrain_print_interval = max(int(getattr(args, "surrogate_pretrain_print_interval", 1)), 1)
    args.surrogate_pretrain_checkpoint = bool(getattr(args, "surrogate_pretrain_checkpoint", True))
    # データセット別Surrogate stateを再利用し、SparsePCGC teacherを毎回ゼロから回す負担を避ける。
    args.surrogate_pretrain_resume = bool(getattr(args, "surrogate_pretrain_resume", True))
    args.surrogate_pretrain_skip_if_loaded = bool(getattr(args, "surrogate_pretrain_skip_if_loaded", False))
    args.surrogate_pretrain_force_retrain = bool(getattr(args, "surrogate_pretrain_force_retrain", False))
    args.surrogate_pretrain_cache_dir = str(getattr(args, "surrogate_pretrain_cache_dir", "")).strip()
    args.surrogate_registry_enabled = bool(getattr(args, "surrogate_registry_enabled", True)) # 共有Surrogate重みの保存/読込を有効化する
    args.surrogate_pretrained_root = str(getattr(args, "surrogate_pretrained_root", str((_DATA_ROOT / "pretrained_surrogate").resolve()))).strip() # 共有Surrogate rootを文字列化する
    if not str(getattr(args, "surrogate_data", "")).strip():
        args.surrogate_data = f"{getattr(args, 'dataname', 'data')}_{getattr(args, 'dataset_name', 'set')}" # 保存ファイル名用のデータ名を既定値で補完する
    else:
        args.surrogate_data = str(getattr(args, "surrogate_data", "")).strip() # 明示指定されたデータ名を使う
    args.surrogate_date = str(getattr(args, "surrogate_date", surrogate_date)).strip() or str(getattr(args, "date", "run")).strip() # 読込対象Surrogateの日付を正規化する
    args.surrogate_time = str(getattr(args, "surrogate_time", surrogate_time)).strip() or str(getattr(args, "time", "run")).strip() # 読込対象Surrogateの時刻を正規化する
    args.surrogate_registry_load_latest_if_missing = bool(getattr(args, "surrogate_registry_load_latest_if_missing", False)) # 指定Surrogateが無い時の最新読込可否を設定する
    args.surrogate_pretrain_legacy_cache_fallback = bool(getattr(args, "surrogate_pretrain_legacy_cache_fallback", False)) # 旧fingerprint cacheを読むかどうかを設定する
    args.surrogate_pretrain_use_replay = bool(getattr(args, "surrogate_pretrain_use_replay", True))
    args.surrogate_pretrain_replay_batch_size = max(
        int(getattr(args, "surrogate_pretrain_replay_batch_size", 16)),
        1,
    )
    args.surrogate_pretrain_replay_steps = max(
        int(getattr(args, "surrogate_pretrain_replay_steps", 4)),
        0,
    )
    replay_update_per_step = getattr(args, "surrogate_pretrain_replay_update_per_step", None)
    if replay_update_per_step is not None:
        args.surrogate_pretrain_replay_steps = max(int(replay_update_per_step), 0)
    args.surrogate_pretrain_replay_update_per_step = int(args.surrogate_pretrain_replay_steps)
    args.surrogate_pretrain_replay_buffer_size = max(
        int(getattr(args, "surrogate_pretrain_replay_buffer_size", 256)),
        0,
    )
    args.surrogate_pretrain_replay_min_size = max(
        int(getattr(args, "surrogate_pretrain_replay_min_size", 4)),
        0,
    )
    args.surrogate_pretrain_allow_stale_target = bool(
        getattr(args, "surrogate_pretrain_allow_stale_target", True)
    )
    args.surrogate_pretrain_max_target_age = max(
        int(getattr(args, "surrogate_pretrain_max_target_age", 20)),
        0,
    )
    args.surrogate_pretrain_skip_on_target_miss = bool(
        getattr(args, "surrogate_pretrain_skip_on_target_miss", False)
    )
    args.surrogate_pretrain_sparsepcgc_debug_interval = int(
        getattr(args, "surrogate_pretrain_sparsepcgc_debug_interval", 10)
    )
    args.surrogate_pretrain_mode = str(getattr(args, "surrogate_pretrain_mode", "full")).strip().lower()
    if args.surrogate_pretrain_mode not in {"full", "subtree", "hybrid"}:
        raise ValueError("--surrogate_pretrain_mode must be one of: full, subtree, hybrid")
    args.surrogate_pretrain_subtree_stub_only = bool(
        getattr(args, "surrogate_pretrain_subtree_stub_only", False)
    )
    args.surrogate_pretrain_subtree_depth_min = int(
        getattr(args, "surrogate_pretrain_subtree_depth_min", -1)
    )
    args.surrogate_pretrain_subtree_depth_max = int(
        getattr(args, "surrogate_pretrain_subtree_depth_max", -1)
    )
    if args.surrogate_pretrain_subtree_depth_min < -1:
        raise ValueError("--surrogate_pretrain_subtree_depth_min must be >= -1")
    if args.surrogate_pretrain_subtree_depth_max < -1:
        raise ValueError("--surrogate_pretrain_subtree_depth_max must be >= -1")
    if (
        args.surrogate_pretrain_subtree_depth_min > 0
        and args.surrogate_pretrain_subtree_depth_max > 0
        and args.surrogate_pretrain_subtree_depth_min > args.surrogate_pretrain_subtree_depth_max
    ):
        args.surrogate_pretrain_subtree_depth_min, args.surrogate_pretrain_subtree_depth_max = (
            args.surrogate_pretrain_subtree_depth_max,
            args.surrogate_pretrain_subtree_depth_min,
        )
    args.surrogate_pretrain_subtree_depth_percent_min = min(
        max(float(getattr(args, "surrogate_pretrain_subtree_depth_percent_min", 0.0)), 0.0),
        1.0,
    )
    args.surrogate_pretrain_subtree_depth_percent_max = min(
        max(float(getattr(args, "surrogate_pretrain_subtree_depth_percent_max", 0.50)), 0.0),
        1.0,
    )
    if args.surrogate_pretrain_subtree_depth_percent_min > args.surrogate_pretrain_subtree_depth_percent_max:
        (
            args.surrogate_pretrain_subtree_depth_percent_min,
            args.surrogate_pretrain_subtree_depth_percent_max,
        ) = (
            args.surrogate_pretrain_subtree_depth_percent_max,
            args.surrogate_pretrain_subtree_depth_percent_min,
        )
    args.surrogate_pretrain_subtree_random_depth = bool(
        getattr(args, "surrogate_pretrain_subtree_random_depth", True)
    )
    args.surrogate_pretrain_subtree_reuse_train_sampler = bool(
        getattr(args, "surrogate_pretrain_subtree_reuse_train_sampler", True)
    )
    args.surrogate_pretrain_subtree_steps_per_full = max(
        int(getattr(args, "surrogate_pretrain_subtree_steps_per_full", 50)),
        1,
    )
    args.surrogate_pretrain_full_calibration_interval = max(
        int(getattr(args, "surrogate_pretrain_full_calibration_interval", 50)),
        1,
    )
    if (
        _cli_option_was_provided("--surrogate_pretrain_subtree_steps_per_full")
        and not _cli_option_was_provided("--surrogate_pretrain_full_calibration_interval")
    ):
        args.surrogate_pretrain_full_calibration_interval = int(args.surrogate_pretrain_subtree_steps_per_full)
    args.surrogate_pretrain_full_calibration_steps = max(
        int(getattr(args, "surrogate_pretrain_full_calibration_steps", 1)),
        1,
    )
    args.surrogate_pretrain_use_full_teacher_for_subtree = bool(
        getattr(args, "surrogate_pretrain_use_full_teacher_for_subtree", False)
    )
    args.surrogate_pretrain_subtree_teacher_type = str(
        getattr(args, "surrogate_pretrain_subtree_teacher_type", "local_actual")
    ).strip().lower()
    if args.surrogate_pretrain_subtree_teacher_type not in {"local_actual", "local_proxy", "inherited_full", "none"}:
        raise ValueError(
            "--surrogate_pretrain_subtree_teacher_type must be one of: "
            "local_actual, local_proxy, inherited_full, none"
        )
    args.surrogate_pretrain_subtree_log_detail = bool(
        getattr(args, "surrogate_pretrain_subtree_log_detail", True)
    )
    args.surrogate_pretrain_max_wall_time_sec = max(
        float(getattr(args, "surrogate_pretrain_max_wall_time_sec", 0.0)),
        0.0,
    )
    args.loss_grad_probe_enabled = bool(getattr(args, "loss_grad_probe_enabled", False))
    args.loss_grad_probe_interval = max(int(getattr(args, "loss_grad_probe_interval", 1)), 0)
    args.step_grad_log = bool(getattr(args, "step_grad_log", True))
    args.step_grad_log_interval = max(int(getattr(args, "step_grad_log_interval", 1)), 1)
    args.step_grad_first_step_only = bool(getattr(args, "step_grad_first_step_only", True))
    args.skip_optimizer_on_nonfinite_grad = bool(getattr(args, "skip_optimizer_on_nonfinite_grad", True))
    args.nonfinite_grad_log_param_limit = max(int(getattr(args, "nonfinite_grad_log_param_limit", 8)), 0)
    args.train_grad_clip = max(float(getattr(args, "train_grad_clip", 0.0)), 0.0)
    args.surrogate_update_during_training = bool(getattr(args, "surrogate_update_during_training", True))
    args.surrogate_update_interval = max(int(getattr(args, "surrogate_update_interval", 1)), 1)
    args.surrogate_joint_lr_scale = max(float(getattr(args, "surrogate_joint_lr_scale", 0.1)), 0.0)
    args.surrogate_update_on_teacher_refresh_only = bool(
        getattr(args, "surrogate_update_on_teacher_refresh_only", False)
    )
    args.surrogate_realign_on_low_corr = bool(getattr(args, "surrogate_realign_on_low_corr", False))
    args.surrogate_realign_min_corr = float(getattr(args, "surrogate_realign_min_corr", 0.3))
    args.surrogate_realign_steps = max(int(getattr(args, "surrogate_realign_steps", 0)), 0)
    args.surrogate_auto_freeze = bool(getattr(args, "surrogate_auto_freeze", False))
    args.surrogate_freeze_abs_error = max(float(getattr(args, "surrogate_freeze_abs_error", 1.0)), 0.0)
    args.surrogate_freeze_train_loss = max(float(getattr(args, "surrogate_freeze_train_loss", 1.0)), 0.0)
    args.surrogate_freeze_patience = max(int(getattr(args, "surrogate_freeze_patience", 8)), 1)
    args.surrogate_resume_abs_error = max(float(getattr(args, "surrogate_resume_abs_error", 2.0)), 0.0)
    args.surrogate_resume_train_loss = max(float(getattr(args, "surrogate_resume_train_loss", 2.0)), 0.0)
    args.compression_good_step_boost = bool(getattr(args, "compression_good_step_boost", True))
    args.compression_good_step_boost_scale = min(
        max(float(getattr(args, "compression_good_step_boost_scale", 1.5)), 1.0),
        4.0,
    )
    args.compression_good_step_prefreeze_scale = min(
        max(float(getattr(args, "compression_good_step_prefreeze_scale", 1.15)), 1.0),
        2.0,
    )
    args.compression_good_step_prefreeze_max_train_loss = max(
        float(getattr(args, "compression_good_step_prefreeze_max_train_loss", 4.0)),
        0.0,
    )
    args.compression_good_step_extra_surrogate_steps = max(
        int(getattr(args, "compression_good_step_extra_surrogate_steps", 4)),
        0,
    )
    args.compression_bad_step_penalty_scale = min(
        max(float(getattr(args, "compression_bad_step_penalty_scale", 1.25)), 1.0),
        4.0,
    )
    args.compression_boost_requires_surrogate_frozen = bool(
        getattr(args, "compression_boost_requires_surrogate_frozen", True)
    )
    args.compression_boost_max_abs_error = max(float(getattr(args, "compression_boost_max_abs_error", 1.0)), 0.0)
    args.compression_surrogate_forward_mode = str(
        getattr(args, "compression_surrogate_forward_mode", "teacher_ste")
    ).strip().lower()
    if args.compression_surrogate_forward_mode not in {"surrogate", "teacher_ste"}:
        raise ValueError("--compression_surrogate_forward_mode must be surrogate or teacher_ste")
    # 出力ノイズを明示的に無効化し、cleanな出力点群を圧縮損失へ渡す既定にする。
    args.disable_output_noise = bool(getattr(args, "disable_output_noise", True))
    # disable_output_noise=Trueなら既存の一様ノイズflagも強制的に無効化する。
    args.use_uniform_noise = bool(getattr(args, "use_uniform_noise", False)) and not args.disable_output_noise
    # actual/surrogate圧縮目的で実bit直結項と内訳項を混ぜる比率を0-1へ収める。
    args.actual_total_bit_objective_mix = min(max(float(getattr(args, "actual_total_bit_objective_mix", 0.5)), 0.0), 1.0)
    # Add/Adjustの実行量をActuator特徴から学習するかをboolへ正規化する。
    args.repair_learn_operation_amounts = bool(getattr(args, "repair_learn_operation_amounts", True))

    # 学習された操作量を位置logitへ戻す強さを非負値へ正規化する。
    args.repair_operation_amount_bias_scale = max(
        float(getattr(args, "repair_operation_amount_bias_scale", 2.0)),
        0.0,
    )

    # Amount ratioから下流の点操作へ向かう勾配だけを強める倍率を正規化する。
    args.repair_amount_downstream_grad_max_scale = max(
        float(getattr(args, "repair_amount_downstream_grad_max_scale", 8.0)),
        1.0,
    )
    args.repair_amount_downstream_grad_scale = max(
        float(getattr(args, "repair_amount_downstream_grad_scale", 6.0)),
        1.0,
    )
    args.repair_drop_amount_downstream_grad_scale = max(
        float(getattr(args, "repair_drop_amount_downstream_grad_scale", args.repair_amount_downstream_grad_scale)),
        1.0,
    )
    args.repair_add_amount_downstream_grad_scale = max(
        float(getattr(args, "repair_add_amount_downstream_grad_scale", args.repair_amount_downstream_grad_scale)),
        1.0,
    )
    args.repair_move_amount_downstream_grad_scale = max(
        float(getattr(args, "repair_move_amount_downstream_grad_scale", args.repair_amount_downstream_grad_scale)),
        1.0,
    )
    args.repair_amount_downstream_grad_scale = min(args.repair_amount_downstream_grad_scale, args.repair_amount_downstream_grad_max_scale)
    args.repair_drop_amount_downstream_grad_scale = min(args.repair_drop_amount_downstream_grad_scale, args.repair_amount_downstream_grad_max_scale)
    args.repair_add_amount_downstream_grad_scale = min(args.repair_add_amount_downstream_grad_scale, args.repair_amount_downstream_grad_max_scale)
    args.repair_move_amount_downstream_grad_scale = min(args.repair_move_amount_downstream_grad_scale, args.repair_amount_downstream_grad_max_scale)
    args.repair_soft_normalizer_floor = max(float(getattr(args, "repair_soft_normalizer_floor", 1e-4)), 1e-8)

    # 学習済み操作割合と実soft操作率を結ぶ補助損失の強さを非負値へ収める。
    args.repair_operation_amount_consistency_weight = max(
        float(getattr(args, "repair_operation_amount_consistency_weight", 1.0)),
        0.0,
    )
    args.repair_operation_amount_direct_weight = max(
        float(getattr(args, "repair_operation_amount_direct_weight", 0.01)),
        0.0,
    )
    args.repair_drop_amount_supervision_weight = max(float(getattr(args, "repair_drop_amount_supervision_weight", 0.001)), 0.0)
    args.repair_drop_amount_soft_consistency_weight = max(float(getattr(args, "repair_drop_amount_soft_consistency_weight", 0.0005)), 0.0)
    args.repair_move_amount_supervision_weight = max(float(getattr(args, "repair_move_amount_supervision_weight", 0.001)), 0.0)
    args.repair_move_amount_soft_consistency_weight = max(float(getattr(args, "repair_move_amount_soft_consistency_weight", 0.0005)), 0.0)
    args.repair_add_amount_supervision_weight = max(float(getattr(args, "repair_add_amount_supervision_weight", 0.001)), 0.0)
    args.repair_add_amount_soft_consistency_weight = max(float(getattr(args, "repair_add_amount_soft_consistency_weight", 0.0005)), 0.0)
    # 学習初期のPrune/Add/Adjust実行量探索率を0-1へ収める。
    args.repair_drop_amount_random_mix_start = min(max(float(getattr(args, "repair_drop_amount_random_mix_start", 0.35)), 0.0), 1.0)
    args.repair_drop_amount_random_mix_end = min(max(float(getattr(args, "repair_drop_amount_random_mix_end", 0.0)), 0.0), 1.0)
    args.repair_add_amount_random_mix_start = min(max(float(getattr(args, "repair_add_amount_random_mix_start", 0.35)), 0.0), 1.0)
    args.repair_add_amount_random_mix_end = min(max(float(getattr(args, "repair_add_amount_random_mix_end", 0.0)), 0.0), 1.0)
    args.repair_move_amount_random_mix_start = min(max(float(getattr(args, "repair_move_amount_random_mix_start", 0.35)), 0.0), 1.0)
    args.repair_move_amount_random_mix_end = min(max(float(getattr(args, "repair_move_amount_random_mix_end", 0.0)), 0.0), 1.0)
    # Adjust source scoreへ入れる探索ノイズ量を非負値へ正規化する。
    args.repair_move_score_noise_start = max(float(getattr(args, "repair_move_score_noise_start", 0.0)), 0.0)
    args.repair_move_score_noise_end = max(float(getattr(args, "repair_move_score_noise_end", 0.0)), 0.0)
    args.noise_delta = float(getattr(args, "noise_delta", 1.0))
    args.log_step_time = bool(getattr(args, "log_step_time", True))
    args.log_gpu_memory = bool(getattr(args, "log_gpu_memory", True))
    args.profile_interval = max(int(getattr(args, "profile_interval", 100)), 1)
    args.lr_scheduler_enabled = bool(getattr(args, "lr_scheduler_enabled", False))
    args.min_main_lr = max(float(getattr(args, "min_main_lr", 1e-5)), 0.0)
    args.min_surrogate_lr = max(float(getattr(args, "min_surrogate_lr", 1e-6)), 0.0)
    # SparsePCGC hard統計は重いため、既定ではprofile間隔と同じ頻度に制限する。
    args.sparsepcgc_hard_debug_interval = max(int(getattr(args, "sparsepcgc_hard_debug_interval", args.profile_interval)), 0)
    # 通常ログにhard統計を連動させるかをboolへ正規化する。
    args.sparsepcgc_hard_debug_on_log = bool(getattr(args, "sparsepcgc_hard_debug_on_log", False))
    args.actual_eval_interval = max(int(getattr(args, "actual_eval_interval", 1000)), 0)
    args.disable_actual_codec_during_train = bool(getattr(args, "disable_actual_codec_during_train", False))
    args.actual_codec_fallback_to_proxy_on_error = bool(getattr(args, "actual_codec_fallback_to_proxy_on_error", True))
    args.skip_optimizer_on_actual_fallback = bool(getattr(args, "skip_optimizer_on_actual_fallback", True))
    args.actual_compression_guard = bool(getattr(args, "actual_compression_guard", True))
    args.actual_guard_patience = max(int(getattr(args, "actual_guard_patience", 2)), 1)
    args.actual_guard_tolerance = max(float(getattr(args, "actual_guard_tolerance", 0.25)), 0.0)
    args.actual_guard_decay_lr = bool(getattr(args, "actual_guard_decay_lr", False))
    args.actual_guard_lr_decay = min(max(float(getattr(args, "actual_guard_lr_decay", 0.5)), 0.0), 1.0)
    args.actual_guard_min_fresh = max(int(getattr(args, "actual_guard_min_fresh", 1)), 1)
    args.actual_guard_restore_best = bool(getattr(args, "actual_guard_restore_best", True))
    args.actual_guard_improvement_epsilon = max(float(getattr(args, "actual_guard_improvement_epsilon", 1e-6)), 0.0)
    args.max_train_steps = max(int(getattr(args, "max_train_steps", 0)), 0)
    args.save_good_bad_cases = bool(getattr(args, "save_good_bad_cases", False))
    args.save_proxy_actual_bad_cases = bool(getattr(args, "save_proxy_actual_bad_cases", True))
    args.proxy_actual_bad_case_threshold = max(float(getattr(args, "proxy_actual_bad_case_threshold", 0.0)), 0.0)
    args.good_case_delta_threshold = float(getattr(args, "good_case_delta_threshold", -5.0))
    args.bad_case_delta_threshold = float(getattr(args, "bad_case_delta_threshold", 20.0))
    args.max_saved_cases = max(int(getattr(args, "max_saved_cases", 64)), 0)
    args.save_case_pointclouds = bool(getattr(args, "save_case_pointclouds", False))
    args.operation_dead_grad_warn_threshold = max(float(getattr(args, "operation_dead_grad_warn_threshold", 1e-12)), 0.0)
    args.operation_dead_grad_warn_patience = max(int(getattr(args, "operation_dead_grad_warn_patience", 20)), 1)
    args.repair_add_ratio_floor = min(max(float(getattr(args, "repair_add_ratio_floor", 0.0)), 0.0), 0.05)
    args.skip_actual_codec = bool(getattr(args, "skip_actual_codec", True))
    args.codec_eval_interval = max(int(getattr(args, "codec_eval_interval", 0)), 0)
    args.profile_test = bool(getattr(args, "profile_test", True))
    args.save_test_ply = bool(getattr(args, "save_test_ply", True))
    args.max_test_samples = max(int(getattr(args, "max_test_samples", 0)), 0)
    args.ply_loader = str(getattr(args, "ply_loader", "numpy")).strip().lower()
    if args.ply_loader == "np":
        args.ply_loader = "numpy"
    elif args.ply_loader == "o3d":
        args.ply_loader = "open3d"
    if args.ply_loader not in {"numpy", "open3d", "auto"}:
        raise ValueError("--ply_loader must be one of: numpy, open3d, auto")
    args.compression_surrogate_aux_node_weight = max(
        float(getattr(args, "compression_surrogate_aux_node_weight", 0.0)),
        0.0,
    )
    args.compression_surrogate_aux_single_weight = max(
        float(getattr(args, "compression_surrogate_aux_single_weight", 0.0)),
        0.0,
    )
    args.compression_surrogate_aux_in_objective = bool(
        getattr(args, "compression_surrogate_aux_in_objective", False)
    )
    args.compression_surrogate_log_soft_aux = bool(
        getattr(args, "compression_surrogate_log_soft_aux", True)
    )
    args.com_sparsepcgc = max(float(getattr(args, "com_sparsepcgc", 0.0)), 0.0)
    args.sparsepcgc_aux_loss = bool(getattr(args, "sparsepcgc_aux_loss", True))
    args.sparsepcgc_aux_backprop = bool(getattr(args, "sparsepcgc_aux_backprop", False))
    args.sparsepcgc_active_coord_weight = max(float(getattr(args, "sparsepcgc_active_coord_weight", 0.60)), 0.0)
    args.sparsepcgc_isolated_proxy_weight = max(float(getattr(args, "sparsepcgc_isolated_proxy_weight", 0.25)), 0.0)
    args.sparsepcgc_entropy_proxy_weight = max(float(getattr(args, "sparsepcgc_entropy_proxy_weight", 0.15)), 0.0)
    args.sparsepcgc_density_proxy_weight = max(float(getattr(args, "sparsepcgc_density_proxy_weight", 0.05)), 0.0)
    args.sparsepcgc_aux_reward_clip = max(float(getattr(args, "sparsepcgc_aux_reward_clip", 0.0)), 0.0)
    args.sparsepcgc_corr_window = max(int(getattr(args, "sparsepcgc_corr_window", 100)), 2)
    args.sparsepcgc_aux_gating = bool(getattr(args, "sparsepcgc_aux_gating", True))
    args.sparsepcgc_aux_min_corr = float(getattr(args, "sparsepcgc_aux_min_corr", 0.30))
    args.sparsepcgc_aux_min_sign_match = min(max(float(getattr(args, "sparsepcgc_aux_min_sign_match", 0.50)), 0.0), 1.0)
    args.sparsepcgc_aux_gating_window = max(int(getattr(args, "sparsepcgc_aux_gating_window", 100)), 2)
    args.sparsepcgc_disable_add = bool(getattr(args, "sparsepcgc_disable_add", True))
    args.surrogate_full_cloud_calib_interval = max(int(getattr(args, "surrogate_full_cloud_calib_interval", 0)), 0)
    args.surrogate_full_cloud_calib_max_samples = max(int(getattr(args, "surrogate_full_cloud_calib_max_samples", 1)), 1)
    args.sparsepcgc_enable_add_experiment = bool(getattr(args, "sparsepcgc_enable_add_experiment", False))
    args.sparsepcgc_add_only_when_compression_primary = bool(
        getattr(args, "sparsepcgc_add_only_when_compression_primary", True)
    )
    args.sparsepcgc_add_target_ratio = min(
        max(float(getattr(args, "sparsepcgc_add_target_ratio", 0.005)), 0.0),
        0.10,
    )
    args.sparsepcgc_add_max_ratio = min(
        max(float(getattr(args, "sparsepcgc_add_max_ratio", 0.10)), args.sparsepcgc_add_target_ratio),
        0.10,
    )
    args.sparsepcgc_add_warmup_steps = max(int(getattr(args, "sparsepcgc_add_warmup_steps", 0)), 0)
    args.sparsepcgc_add_use_candidate_score = bool(getattr(args, "sparsepcgc_add_use_candidate_score", True))
    args.sparsepcgc_add_log_candidates = bool(getattr(args, "sparsepcgc_add_log_candidates", True))
    args.sparsepcgc_add_active_coord_safety_gate = bool(
        getattr(args, "sparsepcgc_add_active_coord_safety_gate", True)
    )
    args.sparsepcgc_add_unique_coord_safety_gate = bool(
        getattr(args, "sparsepcgc_add_unique_coord_safety_gate", True)
    )
    args.sparsepcgc_move_existing_target_only = bool(getattr(args, "sparsepcgc_move_existing_target_only", False))
    args.compression_octree_stat_depth = max(int(getattr(args, "compression_octree_stat_depth", 0)), 0)
    args.compression_octree_stat_force = bool(getattr(args, "compression_octree_stat_force", True))
    sparsepcgc_backend = compress_key == "sparsepcgc" or args.compression_loss_backend.startswith("sparsepcgc_")
    if sparsepcgc_backend:
        if not _cli_option_was_provided("--surrogate_step"):
            args.surrogate_step = max(int(getattr(args, "surrogate_step", 0)), 0)
        if args.compression_loss_backend.endswith("_surrogate") and not _cli_option_was_provided("--disable_actual_codec_during_train"):
            # SparsePCGC surrogateは実SparsePCGC bit教師が必要なので、既定ではactual teacherを止めない。
            args.disable_actual_codec_during_train = False
        if args.compression_loss_backend.endswith("_surrogate") and not _cli_option_was_provided("--surrogate_update_on_teacher_refresh_only"):
            # replay更新も許可し、teacher refresh以外のStepでもSurrogateを継続fitさせる。
            args.surrogate_update_on_teacher_refresh_only = False
        if args.compression_loss_backend.endswith("_surrogate") and not _cli_option_was_provided("--compression_surrogate_replay_steps"):
            # fresh teacherがないStepでもreplay教師で模倣学習を進める。
            args.compression_surrogate_replay_steps = max(int(getattr(args, "compression_surrogate_replay_steps", 0)), 2)
        if args.compression_loss_backend.endswith("_surrogate") and not _cli_option_was_provided("--surrogate_pretrain_actual_refresh_interval"):
            # 事前学習中はSparsePCGC teacherを毎Step更新し、ゼロ/ stale教師だけの学習を避ける。
            args.surrogate_pretrain_actual_refresh_interval = 1
        if not _cli_option_was_provided("--compression_surrogate_forward_mode"):
            args.compression_surrogate_forward_mode = "teacher_ste"
        if not _cli_option_was_provided("--compression_surrogate_refresh_interval"):
            # Surrogate lossは毎Step使うが、実SparsePCGC教師の再計測は高コストなので既定では8Step間隔にする。
            args.compression_surrogate_refresh_interval = 1 if args.compression_loss_backend.endswith("_surrogate") else max(int(getattr(args, "compression_surrogate_refresh_interval", 0)), 50)
        if not _cli_option_was_provided("--lr_scheduler_enabled"):
            # ActualCompressionGuardとStepLRの二重LR低下を避けるため、SparsePCGC実験ではStepLRを既定で止める。
            args.lr_scheduler_enabled = False
        if not _cli_option_was_provided("--compression_surrogate_reuse_last_target"):
            args.compression_surrogate_reuse_last_target = True
        if not _cli_option_was_provided("--train_subtree_anchor_on_min_points_miss"):
            args.train_subtree_anchor_on_min_points_miss = True
        if not _cli_option_was_provided("--compression_surrogate_train_steps"):
            args.compression_surrogate_train_steps = max(args.compression_surrogate_train_steps, 4)
        if not _cli_option_was_provided("--compression_surrogate_warmup_steps"):
            args.compression_surrogate_warmup_steps = max(args.compression_surrogate_warmup_steps, 4)
        if not _cli_option_was_provided("--compression_surrogate_aux_node_weight"):
            args.compression_surrogate_aux_node_weight = 0.0
        if not _cli_option_was_provided("--compression_surrogate_aux_single_weight"):
            args.compression_surrogate_aux_single_weight = 0.0
        if not _cli_option_was_provided("--repair_move_require_empty_target"):
            args.repair_move_require_empty_target = True
        if not _cli_option_was_provided("--repair_move_prefer_occupied_target"):
            args.repair_move_prefer_occupied_target = False
        if not _cli_option_was_provided("--com_sparsepcgc"):
            # SparsePCGC proxyは値と相関検証用に計算するが、backwardへ混ぜるかはgatingで別途決める。
            args.com_sparsepcgc = max(float(getattr(args, "com_sparsepcgc", 0.0)), 0.75)
        if not _cli_option_was_provided("--sparsepcgc_aux_backprop"):
            # actual bitと符号一致していないproxy勾配を混ぜないため、既定ではlog-onlyにする。
            args.sparsepcgc_aux_backprop = False
        if not _cli_option_was_provided("--surrogate_full_cloud_calib_interval"):
            # subtree teacherとfull-cloud actualのズレを定期的に見られるよう、軽い校正anchorを入れる。
            args.surrogate_full_cloud_calib_interval = 200 if args.compression_loss_backend.endswith("_surrogate") else 0
        if not _cli_option_was_provided("--repair_add_ratio_floor"):
            # SparsePCGCでAdd branchが完全に死ぬのを避けるため、0.1%だけ弱い探索下限を入れる。
            args.repair_add_ratio_floor = 0.001
        if not _cli_option_was_provided("--actual_total_bit_objective_mix"):
            args.actual_total_bit_objective_mix = 1.0
        if not _cli_option_was_provided("--compression_boost_requires_surrogate_frozen"):
            args.compression_boost_requires_surrogate_frozen = False
        if not _cli_option_was_provided("--compression_bad_step_penalty_scale"):
            args.compression_bad_step_penalty_scale = 2.0
        if not _cli_option_was_provided("--compression_good_step_boost_scale"):
            args.compression_good_step_boost_scale = 2.0
        if not _cli_option_was_provided("--compression_good_step_prefreeze_scale"):
            args.compression_good_step_prefreeze_scale = 1.5
        if not _cli_option_was_provided("--sparsepcgc_active_coord_weight"):
            args.sparsepcgc_active_coord_weight = max(float(getattr(args, "sparsepcgc_active_coord_weight", 0.60)), 1.00)
        if not _cli_option_was_provided("--sparsepcgc_isolated_proxy_weight"):
            args.sparsepcgc_isolated_proxy_weight = max(float(getattr(args, "sparsepcgc_isolated_proxy_weight", 0.25)), 0.35)
        if not _cli_option_was_provided("--sparsepcgc_entropy_proxy_weight"):
            args.sparsepcgc_entropy_proxy_weight = max(float(getattr(args, "sparsepcgc_entropy_proxy_weight", 0.15)), 0.20)
        if not _cli_option_was_provided("--compression_surrogate_levels"):
            args.compression_surrogate_levels = "2,4,6,8,10"
        if not _cli_option_was_provided("--octree_diag_levels"):
            args.octree_diag_levels = "2,4,6,8,10,12"
        if not _cli_option_was_provided("--sparsepcgc_add_target_ratio"):
            args.sparsepcgc_add_target_ratio = min(float(getattr(args, "sparsepcgc_add_target_ratio", 0.005)), 0.003)
        if not _cli_option_was_provided("--sparsepcgc_add_max_ratio"):
            args.sparsepcgc_add_max_ratio = max(float(getattr(args, "sparsepcgc_add_max_ratio", 0.50)), 0.50) # Addの探索上限を0〜50%へ広げる
        add_experiment_active = bool(args.sparsepcgc_enable_add_experiment) and (
            (not bool(args.sparsepcgc_add_only_when_compression_primary))
            or str(getattr(args, "loss_mode", "legacy_total")).strip().lower() == "compression_primary"
        )
        if add_experiment_active:
            # SparsePCGC Add実験は明示flag時だけ既定disableを解除する。
            # actual active/unique deltaは微分不能なので、ここではratio制御とログに留める。
            args.sparsepcgc_disable_add = False
        if not _cli_option_was_provided("--target_add_ratio"):
            args.target_add_ratio = (
                min(max(float(args.sparsepcgc_add_target_ratio), 0.003), 0.02)
                if add_experiment_active
                else 0.0 if args.sparsepcgc_disable_add else min(
                    max(float(getattr(args, "target_add_ratio", 0.0)), float(args.sparsepcgc_add_target_ratio)),
                    0.02,
                )
            )
        if not _cli_option_was_provided("--max_add_ratio"):
            args.max_add_ratio = (
                min(float(args.sparsepcgc_add_max_ratio), 0.50)
                if add_experiment_active
                else 0.0 if args.sparsepcgc_disable_add else min(
                    max(float(getattr(args, "max_add_ratio", 0.0)), float(args.sparsepcgc_add_max_ratio)),
                    0.50,
                )
            )
        if not _cli_option_was_provided("--target_drop_ratio"):
            args.target_drop_ratio = max(float(getattr(args, "target_drop_ratio", 0.0)), 0.05)
        if not _cli_option_was_provided("--max_drop_ratio"):
            args.max_drop_ratio = max(float(getattr(args, "max_drop_ratio", 0.0)), 0.50) # Pruneの探索上限を0〜50%へ広げる
        if not _cli_option_was_provided("--max_repair_ratio"):
            args.max_repair_ratio = max(float(getattr(args, "max_repair_ratio", 0.0)), 1.00) # Adjustのsource候補上限を0〜100%へ広げる
        if not _cli_option_was_provided("--target_move_ratio"):
            args.target_move_ratio = 0.05
        if not _cli_option_was_provided("--max_move_ratio"):
            args.max_move_ratio = min(
                max(float(getattr(args, "target_move_ratio", 0.05)), 0.15),
                0.20,
            ) # SparsePCGCではMove急増がbit/geometryを崩しやすいため実行上限を抑える
        if not _cli_option_was_provided("--repair_move_warmup_steps"):
            args.repair_move_warmup_steps = max(int(getattr(args, "repair_move_warmup_steps", 0)), 600)
        if not _cli_option_was_provided("--repair_amount_downstream_grad_max_scale"):
            args.repair_amount_downstream_grad_max_scale = min(
                max(float(getattr(args, "repair_amount_downstream_grad_max_scale", 8.0)), 1.0),
                8.0,
            )
        if not _cli_option_was_provided("--repair_soft_normalizer_floor"):
            args.repair_soft_normalizer_floor = max(float(getattr(args, "repair_soft_normalizer_floor", 1e-4)), 1e-4)
        if not _cli_option_was_provided("--train_grad_clip"):
            args.train_grad_clip = 100.0
        if not _cli_option_was_provided("--repair_operation_amount_consistency_weight"):
            args.repair_operation_amount_consistency_weight = min(
                max(float(getattr(args, "repair_operation_amount_consistency_weight", 0.01)), 0.0),
                0.01,
            )
        if not _cli_option_was_provided("--repair_operation_amount_direct_weight"):
            args.repair_operation_amount_direct_weight = max(
                float(getattr(args, "repair_operation_amount_direct_weight", 0.01)),
                0.02,
            )
        if not _cli_option_was_provided("--repair_add_ratio_weight"):
            args.repair_add_ratio_weight = max(float(getattr(args, "repair_add_ratio_weight", 4.0)), 8.0)
        if not _cli_option_was_provided("--repair_add_keep_weight"):
            args.repair_add_keep_weight = 0.10
        if not _cli_option_was_provided("--repair_add_weight_mode"):
            args.repair_add_weight_mode = "hard"
        if not _cli_option_was_provided("--repair_exploration_fraction"):
            args.repair_exploration_fraction = 0.35
        if not _cli_option_was_provided("--repair_add_candidate_ratio_start"):
            args.repair_add_candidate_ratio_start = 0.10
        if not _cli_option_was_provided("--repair_add_candidate_ratio_end"):
            args.repair_add_candidate_ratio_end = 0.02
        if not _cli_option_was_provided("--repair_add_score_noise_start"):
            args.repair_add_score_noise_start = 0.50
        if not _cli_option_was_provided("--repair_add_score_noise_end"):
            args.repair_add_score_noise_end = 0.0
        if not _cli_option_was_provided("--repair_add_weight_random_mix_start"):
            args.repair_add_weight_random_mix_start = 0.0 if args.sparsepcgc_disable_add else 0.10
        if not _cli_option_was_provided("--repair_add_weight_random_mix_end"):
            args.repair_add_weight_random_mix_end = 0.0
        if not _cli_option_was_provided("--repair_drop_amount_random_mix_start"):
            args.repair_drop_amount_random_mix_start = 0.10
        if not _cli_option_was_provided("--repair_drop_amount_random_mix_end"):
            args.repair_drop_amount_random_mix_end = 0.0
        if not _cli_option_was_provided("--repair_add_amount_random_mix_start"):
            args.repair_add_amount_random_mix_start = 0.10
        if not _cli_option_was_provided("--repair_add_amount_random_mix_end"):
            args.repair_add_amount_random_mix_end = 0.0
        if not _cli_option_was_provided("--repair_move_amount_random_mix_start"):
            args.repair_move_amount_random_mix_start = 0.10
        if not _cli_option_was_provided("--repair_move_amount_random_mix_end"):
            args.repair_move_amount_random_mix_end = 0.0
        if not _cli_option_was_provided("--repair_move_score_noise_start"):
            args.repair_move_score_noise_start = 0.15
        if not _cli_option_was_provided("--repair_move_score_noise_end"):
            args.repair_move_score_noise_end = 0.0
        if not _cli_option_was_provided("--repair_drop_score_noise_start"):
            args.repair_drop_score_noise_start = 1.0
        if not _cli_option_was_provided("--repair_drop_score_noise_end"):
            args.repair_drop_score_noise_end = 0.0
        if not _cli_option_was_provided("--repair_drop_random_mix_start"):
            args.repair_drop_random_mix_start = 0.65
        if not _cli_option_was_provided("--repair_drop_random_mix_end"):
            args.repair_drop_random_mix_end = 0.0
        if not _cli_option_was_provided("--max_repair_qstep"):
            args.max_repair_qstep = max(float(getattr(args, "max_repair_qstep", 0.0)), 0.55)
        if not _cli_option_was_provided("--train_subtree_random_full_range"):
            args.train_subtree_random_full_range = False
        if not _cli_option_was_provided("--train_subtree_level_sampling"):
            args.train_subtree_level_sampling = "uniform_random"
        if not _cli_option_was_provided("--train_subtree_curriculum_fraction"):
            args.train_subtree_curriculum_fraction = 1.0
        if not _cli_option_was_provided("--train_subtree_min_points"):
            args.train_subtree_min_points = max(int(getattr(args, "train_subtree_min_points", 1)), 4)
        if not _cli_option_was_provided("--train_patch_subset_anchor_interval"):
            args.train_patch_subset_anchor_interval = 0
        if not _cli_option_was_provided("--train_subtree_full_cloud_prob"):
            args.train_subtree_full_cloud_prob = 0.0
        if not _cli_option_was_provided("--compression_grad_probe"):
            args.compression_grad_probe = True
        if not _cli_option_was_provided("--compression_grad_probe_every"):
            args.compression_grad_probe_every = min(max(int(getattr(args, "compression_grad_probe_every", 10)), 1), 8)
        if not _cli_option_was_provided("--debug_grad_flow"):
            args.debug_grad_flow = True
        if not _cli_option_was_provided("--debug_grad_flow_rate"):
            args.debug_grad_flow_rate = 8

    actual_codec_surrogate_backend = args.compression_loss_backend.endswith("_surrogate")
    if actual_codec_surrogate_backend:
        if (not sparsepcgc_backend) and not _cli_option_was_provided("--disable_actual_codec_during_train"):
            # SparsePCGC以外のsurrogateは従来通り重いactual teacherを既定で抑制する。
            args.disable_actual_codec_during_train = True
        if bool(getattr(args, "disable_actual_codec_during_train", False)):
            if not _cli_option_was_provided("--compression_surrogate_replay_steps"):
                args.compression_surrogate_replay_steps = 0
            if not _cli_option_was_provided("--surrogate_update_on_teacher_refresh_only"):
                args.surrogate_update_on_teacher_refresh_only = True
        if not _cli_option_was_provided("--compression_surrogate_forward_mode"):
            args.compression_surrogate_forward_mode = "teacher_ste"
        if (not sparsepcgc_backend) and (not _cli_option_was_provided("--compression_surrogate_refresh_interval")):
            args.compression_surrogate_refresh_interval = int(getattr(args, "actual_eval_interval", 1000))
        if (not sparsepcgc_backend) and (not _cli_option_was_provided("--compression_surrogate_reuse_last_target")):
            args.compression_surrogate_reuse_last_target = True
        if not _cli_option_was_provided("--compression_surrogate_target_cache_entries"):
            args.compression_surrogate_target_cache_entries = 256
        if not _cli_option_was_provided("--compression_surrogate_aux_node_weight"):
            args.compression_surrogate_aux_node_weight = 0.0
        if not _cli_option_was_provided("--compression_surrogate_aux_single_weight"):
            args.compression_surrogate_aux_single_weight = 0.0
        if not _cli_option_was_provided("--compression_surrogate_train_steps"):
            args.compression_surrogate_train_steps = max(args.compression_surrogate_train_steps, 4)
        if not _cli_option_was_provided("--compression_surrogate_warmup_steps"):
            args.compression_surrogate_warmup_steps = max(args.compression_surrogate_warmup_steps, 4)
        if (not sparsepcgc_backend) and not _cli_option_was_provided("--repair_move_require_empty_target"):
            args.repair_move_require_empty_target = False
        if (not sparsepcgc_backend) and not _cli_option_was_provided("--repair_move_prefer_occupied_target"):
            args.repair_move_prefer_occupied_target = True
        if not _cli_option_was_provided("--max_repair_qstep"):
            args.max_repair_qstep = max(float(getattr(args, "max_repair_qstep", 0.0)), 0.55)

    external_codec_backend = (
        compress_key in {"sparsepcgc", "gpcc", "draco"}
        or args.compression_loss_backend.startswith(("sparsepcgc_", "gpcc_", "draco_"))
    )
    if external_codec_backend:
        if not _cli_option_was_provided("--repair_selection_mode"):
            args.repair_selection_mode = "threshold_cap"
        if sparsepcgc_backend:
            if not _cli_option_was_provided("--repair_move_require_empty_target"):
                args.repair_move_require_empty_target = True
            if not _cli_option_was_provided("--repair_move_prefer_occupied_target"):
                args.repair_move_prefer_occupied_target = False
        else:
            if not _cli_option_was_provided("--repair_move_require_empty_target"):
                args.repair_move_require_empty_target = False
            if not _cli_option_was_provided("--repair_move_prefer_occupied_target"):
                args.repair_move_prefer_occupied_target = True
        if not _cli_option_was_provided("--repair_add_hard_threshold"):
            args.repair_add_hard_threshold = 0.35 if sparsepcgc_backend else 0.0
        if not _cli_option_was_provided("--repair_move_hard_threshold"):
            args.repair_move_hard_threshold = 0.20 if sparsepcgc_backend else min(float(getattr(args, "repair_move_hard_threshold", 0.5)), 0.05)
        if not _cli_option_was_provided("--repair_drop_hard_threshold"):
            args.repair_drop_hard_threshold = min(float(getattr(args, "repair_drop_hard_threshold", 0.5)), 0.25)
        if not _cli_option_was_provided("--repair_exploration_fraction"):
            args.repair_exploration_fraction = max(float(getattr(args, "repair_exploration_fraction", 0.0)), 0.10)
        if not _cli_option_was_provided("--repair_drop_score_noise_start"):
            args.repair_drop_score_noise_start = max(float(getattr(args, "repair_drop_score_noise_start", 0.0)), 0.50)
        if not _cli_option_was_provided("--repair_drop_score_noise_end"):
            args.repair_drop_score_noise_end = 0.0
        if not _cli_option_was_provided("--repair_drop_random_mix_start"):
            args.repair_drop_random_mix_start = max(float(getattr(args, "repair_drop_random_mix_start", 0.0)), 0.35)
        if not _cli_option_was_provided("--repair_drop_random_mix_end"):
            args.repair_drop_random_mix_end = 0.0
        if not _cli_option_was_provided("--repair_add_score_noise_start"):
            args.repair_add_score_noise_start = max(float(getattr(args, "repair_add_score_noise_start", 0.0)), 0.25)
        if not _cli_option_was_provided("--repair_add_score_noise_end"):
            args.repair_add_score_noise_end = 0.0
        if (not sparsepcgc_backend) and (not _cli_option_was_provided("--target_drop_ratio")):
            args.target_drop_ratio = min(float(getattr(args, "target_drop_ratio", 0.0)), 0.02)
        if (not sparsepcgc_backend) and (not _cli_option_was_provided("--max_drop_ratio")):
            args.max_drop_ratio = min(float(getattr(args, "max_drop_ratio", 0.0)), 0.05)

    args.encoder_pre_downsample_mode = str(args.encoder_pre_downsample_mode).strip().lower()
    if args.encoder_pre_downsample_mode not in {"voxel"}:
        raise ValueError(
            f"--encoder_pre_downsample_mode must be 'voxel' (got {args.encoder_pre_downsample_mode})"
        )

    args.encoder_feature_propagation = str(args.encoder_feature_propagation).strip().lower()
    if args.encoder_feature_propagation not in {"knn_inverse_distance"}:
        raise ValueError(
            "--encoder_feature_propagation must be 'knn_inverse_distance' "
            f"(got {args.encoder_feature_propagation})"
        )
    args.encoder_feature_propagation_k = max(int(args.encoder_feature_propagation_k), 1)
    args.allow_slow_knn_fallback = bool(getattr(args, "allow_slow_knn_fallback", False))

    args.patch_parallel_mode = str(args.patch_parallel_mode).strip().lower()
    if args.patch_parallel_mode not in {"auto", "fixed", "all"}:
        raise ValueError(
            f"--patch_parallel_mode must be auto/fixed/all (got {args.patch_parallel_mode})"
        )
    args.patch_batch_size = max(int(args.patch_batch_size), 1)

    args.test_inference_mode = str(getattr(args, "test_inference_mode", "auto")).strip().lower()
    if args.test_inference_mode not in {"auto", "full_cloud", "subtree_merge", "patch", "direct", "legacy"}:
        raise ValueError(
            "--test_inference_mode must be one of: auto, full_cloud, subtree_merge, patch, direct, legacy "
            f"(got {args.test_inference_mode})"
        )
    args.test_auto_time_tolerance = min(max(float(getattr(args, "test_auto_time_tolerance", 0.10)), 0.0), 1.0)
    args.test_allow_subtree_merge = bool(getattr(args, "test_allow_subtree_merge", False))
    args.test_subtree_level = int(getattr(args, "test_subtree_level", 0))
    if args.test_subtree_level < 0:
        raise ValueError("--test_subtree_level must be >= 0")
    args.test_subtree_batch_size = max(int(getattr(args, "test_subtree_batch_size", 1)), 1)
    args.test_subtree_min_points = max(
        int(getattr(args, "test_subtree_min_points", getattr(args, "train_subtree_min_points", 1))),
        1,
    )
    args.test_metric_max_points = max(int(getattr(args, "test_metric_max_points", 8192)), 0)
    args.test_metric_normal_k = max(int(getattr(args, "test_metric_normal_k", 16)), 3)
    args.test_compute_quality_metrics = bool(getattr(args, "test_compute_quality_metrics", True))

    args.train_patch_subset_sampling = str(getattr(args, "train_patch_subset_sampling", "coverage_cycle")).strip().lower()
    if args.train_patch_subset_sampling not in {"coverage_cycle"}:
        raise ValueError(
            "--train_patch_subset_sampling must be coverage_cycle "
            f"(got {args.train_patch_subset_sampling})"
        )
    args.train_patch_subset_patches_per_step = int(getattr(args, "train_patch_subset_patches_per_step", 0))
    if args.train_patch_subset_patches_per_step < 1:
        raise ValueError("--train_patch_subset_patches_per_step must be >= 1")
    args.train_patch_subset_anchor_interval = int(getattr(args, "train_patch_subset_anchor_interval", 0))
    if args.train_patch_subset_anchor_interval < 0:
        raise ValueError("--train_patch_subset_anchor_interval must be >= 0")
    args.train_subtree_full_cloud_prob = float(getattr(args, "train_subtree_full_cloud_prob", 0.0))
    if not 0.0 <= args.train_subtree_full_cloud_prob <= 1.0:
        raise ValueError("--train_subtree_full_cloud_prob must be in [0, 1]")
    args.train_subtree_level = int(getattr(args, "train_subtree_level", 0))
    if args.train_subtree_level < 0:
        raise ValueError("--train_subtree_level must be >= 0")
    args.train_subtree_level_jitter = max(int(getattr(args, "train_subtree_level_jitter", 0)), 0)
    args.train_subtree_level_min = int(getattr(args, "train_subtree_level_min", 0))
    args.train_subtree_level_max = int(getattr(args, "train_subtree_level_max", 0))
    if args.train_subtree_level_min < 0:
        raise ValueError("--train_subtree_level_min must be >= 0")
    if args.train_subtree_level_max < 0:
        raise ValueError("--train_subtree_level_max must be >= 0")
    args.train_subtree_level_sampling = str(getattr(args, "train_subtree_level_sampling", "uniform_random")).strip().lower()
    if args.train_subtree_level_sampling not in {"uniform_random", "coverage_cycle"}:
        raise ValueError(
            "--train_subtree_level_sampling must be one of: uniform_random, coverage_cycle "
            f"(got {args.train_subtree_level_sampling})"
        )
    args.train_subtree_level_curriculum = bool(getattr(args, "train_subtree_level_curriculum", True))
    args.train_subtree_curriculum_fraction = min(
        max(float(getattr(args, "train_subtree_curriculum_fraction", 1.0)), 0.0),
        1.0,
    )
    args.train_subtree_curriculum_direction = str(
        getattr(args, "train_subtree_curriculum_direction", "deep_to_shallow")
    ).strip().lower()
    if args.train_subtree_curriculum_direction not in {"deep_to_shallow", "shallow_to_deep"}:
        raise ValueError(
            "--train_subtree_curriculum_direction must be one of: "
            "deep_to_shallow, shallow_to_deep "
            f"(got {args.train_subtree_curriculum_direction})"
        )
    args.train_subtree_depth_percent_curriculum = bool(
        getattr(args, "train_subtree_depth_percent_curriculum", True)
    )
    args.train_subtree_depth_percent_start = _parse_csv_floats(
        getattr(args, "train_subtree_depth_percent_start", "0.0,0.50")
    )
    args.train_subtree_depth_percent_end = _parse_csv_floats(
        getattr(args, "train_subtree_depth_percent_end", "0.0,0.50")
    )
    for label, values in (
        ("--train_subtree_depth_percent_start", args.train_subtree_depth_percent_start),
        ("--train_subtree_depth_percent_end", args.train_subtree_depth_percent_end),
    ):
        if len(values) != 2:
            raise ValueError(f"{label} must contain two comma-separated values, e.g. 0.70,0.90")
        lo, hi = sorted(float(value) for value in values)
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"{label} values must be in [0, 1] (got {values})")
        if hi <= 0.0:
            raise ValueError(f"{label} upper value must be > 0 (got {values})")
    args.train_subtree_depth_percent_start = sorted(args.train_subtree_depth_percent_start)
    args.train_subtree_depth_percent_end = sorted(args.train_subtree_depth_percent_end)
    if not _cli_option_was_provided("--surrogate_pretrain_subtree_depth_percent_min"):
        args.surrogate_pretrain_subtree_depth_percent_min = float(min(args.train_subtree_depth_percent_start + args.train_subtree_depth_percent_end)) # Surrogate事前学習の最小深さ割合を本学習レンジへ揃える
    if not _cli_option_was_provided("--surrogate_pretrain_subtree_depth_percent_max"):
        args.surrogate_pretrain_subtree_depth_percent_max = float(max(args.train_subtree_depth_percent_start + args.train_subtree_depth_percent_end)) # Surrogate事前学習の最大深さ割合を本学習レンジへ揃える
    args._train_subtree_depth_cli_override = bool(
        _cli_option_was_provided("--train_subtree_level")
        or _cli_option_was_provided("--train_subtree_level_min")
        or _cli_option_was_provided("--train_subtree_level_max")
    )
    args.train_subtree_random_full_range = bool(getattr(args, "train_subtree_random_full_range", True))
    args.train_subtree_min_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1)
    args.train_subtree_stat_log_limit = max(int(getattr(args, "train_subtree_stat_log_limit", 16)), 0)
    args.target_add_ratio = min(max(float(getattr(args, "target_add_ratio", 0.0)), 0.0), 0.95)
    args.max_add_ratio = min(max(float(getattr(args, "max_add_ratio", args.target_add_ratio)), args.target_add_ratio), 1.0)
    args.target_drop_ratio = min(max(float(getattr(args, "target_drop_ratio", 0.0)), 0.0), 0.95)
    args.max_drop_ratio = min(max(float(getattr(args, "max_drop_ratio", args.target_drop_ratio)), args.target_drop_ratio), 1.0)
    args.target_repair_ratio = min(max(float(getattr(args, "target_repair_ratio", 0.0)), 0.0), 0.95)
    args.max_repair_ratio = min(max(float(getattr(args, "max_repair_ratio", args.target_repair_ratio)), args.target_repair_ratio), 1.0)
    args.target_move_ratio = min(max(float(getattr(args, "target_move_ratio", args.target_repair_ratio)), 0.0), 0.95)
    args.max_move_ratio = min(max(float(getattr(args, "max_move_ratio", args.target_move_ratio)), args.target_move_ratio), 1.0)
    args.repair_move_ratio_floor = min(
        max(float(getattr(args, "repair_move_ratio_floor", 0.0)), 0.0),
        args.max_move_ratio,
    )
    args.repair_move_warmup_steps = max(int(getattr(args, "repair_move_warmup_steps", 0)), 0)
    args.repair_delete_max_points_per_voxel = max(int(getattr(args, "repair_delete_max_points_per_voxel", 8)), 0)
    args.repair_move_max_points_per_voxel = max(int(getattr(args, "repair_move_max_points_per_voxel", 8)), 0)
    args.add_noop_keep_threshold = min(max(float(getattr(args, "add_noop_keep_threshold", 0.5)), 0.0), 1.0)
    args.repair_add_drop_conflict_weight = max(float(getattr(args, "repair_add_drop_conflict_weight", 0.0)), 0.0)
    args.repair_add_keep_weight = max(float(getattr(args, "repair_add_keep_weight", 0.0)), 0.0)
    args.repair_add_min_offset_qstep = max(float(getattr(args, "repair_add_min_offset_qstep", 0.0)), 0.0)
    args.repair_add_min_offset_weight = max(float(getattr(args, "repair_add_min_offset_weight", 0.0)), 0.0)
    args.repair_move_require_empty_target = bool(getattr(args, "repair_move_require_empty_target", True))
    args.repair_move_prefer_occupied_target = bool(getattr(args, "repair_move_prefer_occupied_target", False))
    args.repair_move_source_prior_weight = min(
        max(float(getattr(args, "repair_move_source_prior_weight", 0.35)), 0.0),
        1.0,
    )
    args.sparsepcgc_move_source_prior_weight = min(
        max(float(getattr(args, "sparsepcgc_move_source_prior_weight", 0.55)), 0.0),
        1.0,
    )
    args.enable_voxel_collision_log = bool(getattr(args, "enable_voxel_collision_log", True))
    args.voxel_collision_log_interval = max(int(getattr(args, "voxel_collision_log_interval", 1)), 1)
    args.voxel_collision_max_points = max(int(getattr(args, "voxel_collision_max_points", 300000)), 0)
    args.voxel_collision_log_first_batch_only = bool(getattr(args, "voxel_collision_log_first_batch_only", True))
    args.voxel_collision_log_stages = str(
        getattr(args, "voxel_collision_log_stages", "input_gt,model_output_raw,compression_input")
    )
    args.enable_sparsepcgc_empty_target_guard = bool(getattr(args, "enable_sparsepcgc_empty_target_guard", True))
    args.enable_sparsepcgc_target_duplicate_guard = bool(getattr(args, "enable_sparsepcgc_target_duplicate_guard", True))
    args.sparsepcgc_empty_target_penalty_weight = max(
        float(getattr(args, "sparsepcgc_empty_target_penalty_weight", 0.0)),
        0.0,
    )
    args.sparsepcgc_target_duplicate_penalty_weight = max(
        float(getattr(args, "sparsepcgc_target_duplicate_penalty_weight", 0.0)),
        0.0,
    )
    args.enable_sparsepcgc_occupancy_debug = bool(getattr(args, "enable_sparsepcgc_occupancy_debug", False))
    args.sparsepcgc_occupancy_low_prob_threshold = min(
        max(float(getattr(args, "sparsepcgc_occupancy_low_prob_threshold", 0.1)), 1e-6),
        1.0,
    )
    args.enable_sparsepcgc_exact_occupancy_teacher = bool(getattr(args, "enable_sparsepcgc_exact_occupancy_teacher", False))
    args.sparsepcgc_exact_occupancy_interval = max(
        int(getattr(args, "sparsepcgc_exact_occupancy_interval", 1)),
        0,
    )
    exact_teacher_mode = str(getattr(args, "sparsepcgc_exact_teacher_mode", "auto")).strip().lower()
    if exact_teacher_mode not in {"auto", "full_cloud", "global_subtree", "local_subtree"}:
        raise ValueError("--sparsepcgc_exact_teacher_mode must be auto/full_cloud/global_subtree/local_subtree")
    args.sparsepcgc_exact_teacher_mode = exact_teacher_mode
    args.enable_sparsepcgc_exact_occupancy_loss = bool(
        getattr(args, "enable_sparsepcgc_exact_occupancy_loss", False)
    )
    args.sparsepcgc_exact_occupancy_loss_weight = max(
        float(getattr(args, "sparsepcgc_exact_occupancy_loss_weight", 0.0)),
        0.0,
    )
    args.sparsepcgc_exact_bits_loss_weight = max(
        float(getattr(args, "sparsepcgc_exact_bits_loss_weight", 0.0)),
        0.0,
    )
    selection_mode = str(getattr(args, "repair_selection_mode", "target")).strip().lower().replace("-", "_")
    if selection_mode in {"cap", "optional", "threshold", "thresholdcap"}:
        selection_mode = "threshold_cap"
    if selection_mode not in {"target", "threshold_cap"}:
        raise ValueError("--repair_selection_mode must be target or threshold_cap")
    args.repair_selection_mode = selection_mode
    args.repair_move_hard_threshold = min(max(float(getattr(args, "repair_move_hard_threshold", 0.5)), 0.0), 1.0)
    args.repair_drop_hard_threshold = min(max(float(getattr(args, "repair_drop_hard_threshold", 0.5)), 0.0), 1.0)
    args.repair_add_hard_threshold = min(max(float(getattr(args, "repair_add_hard_threshold", 0.5)), 0.0), 1.0)
    args.repair_quant_guard_weight = max(float(getattr(args, "repair_quant_guard_weight", 0.0)), 0.0)
    args.repair_local_guard_weight = max(float(getattr(args, "repair_local_guard_weight", 0.0)), 0.0)
    args.repair_drop_soft_proxy_tau = max(float(getattr(args, "repair_drop_soft_proxy_tau", 8.0)), 1e-6)
    args.repair_drop_direct_target_weight = max(float(getattr(args, "repair_drop_direct_target_weight", 5.0)), 0.0)
    args.repair_drop_entropy_weight = max(float(getattr(args, "repair_drop_entropy_weight", 0.01)), 0.0)
    args.repair_add_weight_mode = str(getattr(args, "repair_add_weight_mode", "hard")).strip().lower()
    if args.repair_add_weight_mode not in {"hard", "soft"}:
        raise ValueError("--repair_add_weight_mode must be hard or soft")
    args.repair_exploration_fraction = min(max(float(getattr(args, "repair_exploration_fraction", 0.0)), 0.0), 1.0)
    args.repair_add_candidate_ratio_start = max(float(getattr(args, "repair_add_candidate_ratio_start", 0.0)), 0.0)
    args.repair_add_candidate_ratio_end = max(float(getattr(args, "repair_add_candidate_ratio_end", 0.0)), 0.0)
    args.repair_add_score_noise_start = max(float(getattr(args, "repair_add_score_noise_start", 0.0)), 0.0)
    args.repair_add_score_noise_end = max(float(getattr(args, "repair_add_score_noise_end", 0.0)), 0.0)
    args.repair_add_weight_random_mix_start = min(max(float(getattr(args, "repair_add_weight_random_mix_start", 0.0)), 0.0), 1.0)
    args.repair_add_weight_random_mix_end = min(max(float(getattr(args, "repair_add_weight_random_mix_end", 0.0)), 0.0), 1.0)
    args.repair_drop_score_noise_start = max(float(getattr(args, "repair_drop_score_noise_start", 0.0)), 0.0)
    args.repair_drop_score_noise_end = max(float(getattr(args, "repair_drop_score_noise_end", 0.0)), 0.0)
    args.repair_drop_random_mix_start = min(max(float(getattr(args, "repair_drop_random_mix_start", 0.0)), 0.0), 1.0)
    args.repair_drop_random_mix_end = min(max(float(getattr(args, "repair_drop_random_mix_end", 0.0)), 0.0), 1.0)
    args.allow_local_repair_unit_recompute = bool(getattr(args, "allow_local_repair_unit_recompute", False))
    args.allow_local_octree_recompute = bool(getattr(args, "allow_local_octree_recompute", False))
    args.forbid_local_voxel_recompute = bool(getattr(args, "forbid_local_voxel_recompute", False))
    args.ckpt = _resolve_repo_or_cwd_path(args.ckpt)
    args.octattention_ckpt = _resolve_repo_or_cwd_path(args.octattention_ckpt)
    args.sparsepcgc_root = _resolve_repo_or_cwd_path(args.sparsepcgc_root)
    args.sparsepcgc_ckptdir = _resolve_from_base_path(args.sparsepcgc_ckptdir, args.sparsepcgc_root)
    args.sparsepcgc_ckptdir_sr = _resolve_from_base_path(args.sparsepcgc_ckptdir_sr, args.sparsepcgc_root)
    args.sparsepcgc_ckptdir_ae = _resolve_from_base_path(args.sparsepcgc_ckptdir_ae, args.sparsepcgc_root)
    args.sparsepcgc_ckptdir_low = _resolve_from_base_path(args.sparsepcgc_ckptdir_low, args.sparsepcgc_root)
    args.sparsepcgc_ckptdir_high = _resolve_from_base_path(args.sparsepcgc_ckptdir_high, args.sparsepcgc_root)
    args.sparsepcgc_ckptdir_offset = _resolve_from_base_path(args.sparsepcgc_ckptdir_offset, args.sparsepcgc_root)
    sparse_python = str(getattr(args, "sparsepcgc_python", "")).strip()
    if sparse_python and (os.path.isabs(sparse_python) or os.path.sep in sparse_python):
        args.sparsepcgc_python = _resolve_repo_or_cwd_path(sparse_python)
    else:
        args.sparsepcgc_python = sparse_python
    args.gpcc_root = _resolve_repo_or_cwd_path(args.gpcc_root)
    args.gpcc_encoder_path = _resolve_from_base_path(args.gpcc_encoder_path, args.gpcc_root)
    args.gpcc_cfg_dir = _resolve_from_base_path(args.gpcc_cfg_dir, args.gpcc_root)
    args.draco_root = _resolve_repo_or_cwd_path(args.draco_root)
    args.draco_encoder_path = _resolve_from_base_path(args.draco_encoder_path, args.draco_root)
    args.draco_decoder_path = _resolve_from_base_path(args.draco_decoder_path, args.draco_root)
    args.save_dir = _resolve_repo_or_cwd_path(args.save_dir)
    args.out_path = _resolve_repo_or_cwd_path(args.out_path)
    args.log_root = _resolve_repo_or_cwd_path(args.log_root)
    args.save_ply_dir = _resolve_repo_or_cwd_path(args.save_ply_dir)
    args.codec_eval_dir = _resolve_repo_or_cwd_path(args.codec_eval_dir)
    args.output_log = _resolve_repo_or_cwd_path(args.output_log)

    return args
