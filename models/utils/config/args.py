import argparse
import os
import sys
from pathlib import Path
from cfgs.utils import str2bool

pretrained_date = "20260508"
pretrained_time = "190424"
method_loss = "cd"
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


def _data_subset_dir(split: str, data_name: str = dataname, subset_name: str = dataset_name) -> Path:
    return _DATA_ROOT / split / str(data_name) / str(subset_name)


def _cli_option_was_provided(option_name: str) -> bool:
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in sys.argv[1:])


def _discover_latest_pretrained_checkpoint(model_stem: str = model_name) -> str:
    candidates = []
    candidates.extend(_LOG_ROOT.glob(f"*/MyNetwork_train/checkpoints/*/{model_stem}.pth"))
    candidates.extend(_PRETRAINED_ROOT.glob(f"*/*/{model_stem}.pth"))
    candidates.extend(_LEGACY_PRETRAINED_ROOT.glob(f"*/*/{model_stem}.pth"))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if existing:
        latest = max(existing, key=lambda path: path.stat().st_mtime)
        return str(latest.resolve())
    fallback = _PRETRAINED_ROOT / pretrained_date / f"{pretrained_time}_{method_loss}" / f"{model_stem}.pth"
    return str(fallback.resolve())


def _default_checkpoint_path() -> str:
    preferred = _PRETRAINED_ROOT / pretrained_date / f"{pretrained_time}_{method_loss}" / f"{model_name}.pth"
    if preferred.is_file():
        return str(preferred.resolve())
    data_root_candidates = sorted(_PRETRAINED_ROOT.glob(f"*/*/{model_name}.pth"))
    if data_root_candidates:
        return str(data_root_candidates[-1].resolve())
    legacy_preferred = _LEGACY_PRETRAINED_ROOT / pretrained_date / f"{pretrained_time}_{method_loss}" / f"{model_name}.pth"
    if legacy_preferred.is_file():
        return str(legacy_preferred.resolve())
    return _discover_latest_pretrained_checkpoint(model_stem=model_name)


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


def _compress_key(raw_value: str) -> str:
    return str(raw_value).strip().lower().replace("_", "").replace("-", "")


def _compress_display_name(raw_value: str) -> str:
    key = _compress_key(raw_value)
    if key == "sparsepcgc":
        return "SparsePCGC"
    if key == "gpcc":
        return "G-PCC"
    if key == "octattention":
        return "OctAttention"
    text = str(raw_value).strip()
    return text if text else "OctAttention"


def parse_pugan_args(parser, file_day, file_time):
    """基本情報"""
    parser.add_argument('--date', default=f'{file_day}', type=str, help='日付')
    parser.add_argument('--time', default=f'{file_time}', type=str, help='時刻')
    parser.add_argument('--input_dir', default=str(_DATA_ROOT / "ground"), type=str, help='入力点群データのフォルダパス')
    parser.add_argument('--cpu', action='store_true', help='GPUを使わずCPUで学習するかどうか')
    parser.add_argument('--print_rate', default=1, type=int, help='ログ出力頻度（1なら毎ステップ、0なら最初と最後のみ）')
    parser.add_argument('--dataname', default=dataname, type=str, help='データセットの名称')
    parser.add_argument('--dataset_name', default=dataset_name, type=str, help='データセット内シーケンスの名称')

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
    parser.add_argument('--target_add_ratio', default=0.01, type=float, help='目標とする追加割合')
    parser.add_argument('--max_add_ratio', default=0.03, type=float, help='追加割合の最大値')
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
    parser.add_argument('--repair_ratio_weight', default=8.0, type=float, help='修復割合制御損失の重み')
    parser.add_argument('--repair_shape_guard_weight', default=0.5, type=float, help='形状保持原因が強い点を動かしすぎない正則化')
    parser.add_argument('--target_drop_ratio', default=0.01, type=float, help='点削除ゲートの目標割合')
    parser.add_argument('--max_drop_ratio', default=0.03, type=float, help='点削除ゲートの上限割合')
    parser.add_argument('--repair_drop_ratio_weight', default=4.0, type=float, help='点削除割合制御損失の重み')
    parser.add_argument('--repair_drop_shape_guard_weight', default=1.0, type=float, help='形状保持点を削除しすぎない正則化')
    parser.add_argument('--repair_priority_gate', default=True, type=str2bool, help='高コスト領域だけを修復対象にする優先度ゲートを使うか')
    parser.add_argument('--repair_priority_gate_tau', default=0.08, type=float, help='修復優先度ゲートの温度')
    parser.add_argument('--repair_gate_mean_cap', default=True, type=str2bool, help='修復対象割合の平均がtarget_repair_ratioを超えないように再スケールするか')
    parser.add_argument('--repair_unit_level', default=5, type=int, help='原因集約で使う粗いOctree/subtree単位の深さ')
    parser.add_argument('--structure_geo_k', default=8, type=int, help='構造解析に使う局所幾何kNN数')
    parser.add_argument('--structure_geo_max_points', default=2048, type=int, help='局所幾何統計を厳密計算する最大点数（超過時はOctree特徴を優先）')
    parser.add_argument('--octree_diag_levels', default='4,6,8,10,12', type=str, help='Octree階層ごとの診断ログを出すレベル')
    parser.add_argument('--training_stage', default='joint', type=str, help='学習段階(diagnosis/joint)')
    parser.add_argument('--two_stage_training', default=True, type=str2bool, help='2段階学習(diagnosis->joint)を自動で行うか')
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
    parser.add_argument('--out_path', default=str((_LOG_ROOT / file_day / "MyNetwork_train" / "checkpoints" / file_time).resolve()), type=str, help='チェックポイント保存先')
    parser.add_argument('--log_root', default=str(_LOG_ROOT), type=str, help='学習・推論ログ保存ルート')
    parser.add_argument('--optim', default='adam', type=str, help='最適化手法（adamまたはsgd）')
    parser.add_argument('--expansion', action='store_true', help='拡張データを使用するか')
    parser.add_argument('--gamma', default=0.5, type=float, help='学習率減衰の係数')
    parser.add_argument('--lr_decay_step', default=24, type=int, help='学習率を減衰させるステップ間隔')
    parser.add_argument('--max_files', default=30, type=int, help='読み込む最大ファイル数')
    parser.add_argument('--episodes', default=64, type=int, help='学習エピソード数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学習率')
    parser.add_argument('--save_eval', default='loss', type=str, help='評価指標（lossまたはpsnr）')
    parser.add_argument('--deform', default=False, type=str2bool, help='変形モジュールをゆっくり学習するか')
    parser.add_argument('--loss_type', default='cd', type=str, help='幾何損失の種類')
    parser.add_argument('--method_name', default=method_name, type=str, help='ログ上の提案手法名')
    parser.add_argument('--run_name', default='', type=str, help='チェックポイント保存名。空なら <time>_<compress> を使う')
    parser.add_argument('--geometry_audit_max_points', default=8192, type=int, help='geometry監査用にCDを計算する最大点数(0で無効)')
    parser.add_argument('--operation_count_drop_threshold', default=0.50, type=float, help='学習ログで削除点として数えるkeep確率のしきい値')
    parser.add_argument('--operation_count_adjust_threshold', default=1e-6, type=float, help='学習ログで調整点として数える最小移動距離')

    # 損失関数のパラメータ
    parser.add_argument('--com_bit',    default=10*100, type=float, help='train.pyが最終圧縮損失を合成するときのbit差(%)項の重み')
    parser.add_argument('--com_sin',    default=1, type=float, help='train.pyが最終圧縮損失を合成するときのsingle-child差(%)項の重み')
    parser.add_argument('--com_node',   default=4, type=float, help='train.pyが最終圧縮損失を合成するときのnode数差(%)項の重み')
    parser.add_argument('--com_bpn',    default=0.25, type=float, help='train.pyが最終圧縮損失を合成するときのbits-per-node差(%)項の重み')
    parser.add_argument('--com_lowprob', default=1, type=float, help='train.pyが最終圧縮損失を合成するときのlow-probability occupancy項の重み')
    parser.add_argument('--com_ent',   default=2, type=float, help='エントロピー損失の重み')
    parser.add_argument('--prun_cnt',   default=5, type=float, help='Pruningの個数制御損失')
    parser.add_argument('--prun_out',   default=20*100, type=float, help='Pruningの外れ値損失')
    parser.add_argument('--add_cnt',    default=5, type=float, help='Addの個数制御損失')
    parser.add_argument('--add_fit',    default=4*100, type=float, help='Addのフィッティング損失')
    parser.add_argument('--add_rep',    default=1*100, type=float, help='Addの分散抑制損失')
    parser.add_argument('--disp_cnt',    default=5, type=float, help='Displacementの個数制御損失')
    parser.add_argument('--disp_fit',    default=4*100, type=float, help='Displacementのフィッティング損失')
    parser.add_argument('--w_geom',     default=10**7, type=float, help='幾何損失ブロック全体の重み')
    parser.add_argument('--w_com',      default=10, type=float, help='最終圧縮損失ブロック全体の重み（各com_*項をまとめて掛ける）')
    parser.add_argument('--w_prun',     default=1, type=float, help='原因分解損失ブロック全体の重み（旧Pruning枠）')
    parser.add_argument('--w_add',      default=1, type=float, help='構造修復ポリシー損失ブロック全体の重み（旧Add枠）')
    parser.add_argument('--w_dis',      default=1, type=float, help='構造修復アクチュエータ損失ブロック全体の重み（旧Displacement枠）')

    parser.add_argument('--lambda_p',   default=10**-5, type=float, help='soft圧縮損失の係数')
    parser.add_argument('--discrete_loss_mode', default='ste_hard', type=str, help='離散学習のモード')
    parser.add_argument('--discrete_surrogate_weight', default=1.0, type=float, help='STE時の代理勾配の重み')
    parser.add_argument('--discrete_policy_weight', default=1, type=float, help='ポリシー勾配の重み')
    parser.add_argument('--discrete_policy_reward_clip', default=100.0, type=float, help='報酬のクリップ値（0で無効）')
    parser.add_argument('--discrete_policy_baseline_momentum', default=0.95, type=float, help='ベースラインのEMA係数')

    """Compression"""
    parser.add_argument('--compress', default='OctAttention', type=str, help='使用する圧縮手法')
    parser.add_argument('--octree_voxel', type=float, default=1e-3, help='Octreeボクセルサイズ')
    parser.add_argument('--qs', type=int, default=2, help='量子化ステップサイズ')

    # Octree Compression
    parser.add_argument('--max_gpu_mem_it', type=int, default=2**9, help='GPUメモリ制限に応じた反復回数')
    parser.add_argument('--oa_subprocess', default=False, type=str2bool, help='サブプロセスで圧縮を行うか')
    parser.add_argument('--surrogate', default=True, type=str2bool, help='TrueならproxyではなくOctAttention surrogateを圧縮損失に使う')
    parser.add_argument('--compression_loss_backend', default='proxy', type=str, help='圧縮損失の計算方法(proxy/octattention_actual/octattention_actual_ste/octattention_surrogate/sparsepcgc_actual/sparsepcgc_actual_ste/sparsepcgc_surrogate/gpcc_actual/gpcc_actual_ste/gpcc_surrogate)。surrogateは実圧縮教師の百分率を周期的に模倣する')
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
    parser.add_argument('--sparsepcgc_match_qs', default=True, type=str2bool, help='SparsePCGCの有効量子化幅を--qsに合わせる（明示指定が無い場合）')
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
    parser.add_argument('--compression_surrogate_levels', default='4,6,8', type=str, help='Soft octree surrogate特徴に使う階層')
    parser.add_argument('--compression_surrogate_hidden_dim', default=128, type=int, help='圧縮サロゲートMLPの隠れ次元')
    parser.add_argument('--compression_surrogate_lr', default=3e-3, type=float, help='圧縮サロゲートのオンライン学習率')
    parser.add_argument('--compression_surrogate_weight_decay', default=1e-5, type=float, help='圧縮サロゲートのweight decay')
    parser.add_argument('--compression_surrogate_train_steps', default=2, type=int, help='教師更新時にサロゲートを教師bitに合わせて更新する回数')
    parser.add_argument('--compression_surrogate_warmup_steps', default=2, type=int, help='実ネットワーク更新前に行うSurrogate専用学習回数')
    parser.add_argument('--compression_surrogate_refresh_interval', default=8, type=int, help='何train stepごとに実圧縮教師を再計測するか(0なら初回以外は再計測しない)')
    parser.add_argument('--compression_surrogate_reuse_last_target', default=True, type=str2bool, help='未計測subtreeでは直近の実圧縮教師targetを再利用するか')
    parser.add_argument('--compression_surrogate_target_cache_entries', default=256, type=int, help='Surrogate教師targetのLRUキャッシュ数')
    parser.add_argument('--compression_surrogate_replay_steps', default=1, type=int, help='実圧縮を呼ばないstepでもreplay教師でSurrogateを更新する回数')
    parser.add_argument('--compression_surrogate_replay_batch', default=8, type=int, help='Surrogate replay学習のbatch数')
    parser.add_argument('--compression_surrogate_replay_entries', default=512, type=int, help='Surrogate replay bufferの最大件数')
    parser.add_argument('--compression_surrogate_forward_mode', default='surrogate', type=str, help='surrogate損失のforward値(surrogate/teacher_ste)')
    parser.add_argument('--compression_surrogate_aux_node_weight', default=0.05, type=float, help='Surrogate損失に足すsoft Octree node補助項の重み')
    parser.add_argument('--compression_surrogate_aux_single_weight', default=0.05, type=float, help='Surrogate損失に足すsoft単一子ノード補助項の重み')
    parser.add_argument('--compression_octree_stat_depth', default=0, type=int, help='実圧縮debug用Octree統計の深さ(0なら点群から推定)')
    parser.add_argument('--compression_octree_stat_force', default=True, type=str2bool, help='圧縮器が返すnode/singleが0でも点群からOctree統計を補完する')
    parser.add_argument('--compression_surrogate_grad_clip', default=10.0, type=float, help='圧縮サロゲートの勾配クリップ')
    parser.add_argument('--compression_surrogate_target_scale', default=100.0, type=float, help='旧互換用。実圧縮教師百分率モードでは未使用')
    parser.add_argument('--compression_surrogate_pred_clip', default=100.0, type=float, help='サロゲートが予測する実圧縮bit差百分率のtanhクリップ（0で無効）')
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

    # proxyOctreeCompression
    parser.add_argument('--proxy_max_depth',     default=12,    type=int,   help='Octreeの最大深さ')
    parser.add_argument('--proxy_lambda_entropy', default=1,    type=float,   help='エントロピー項の重み')
    parser.add_argument('--proxy_lambda_node_count',   default=1,  type=float,   help='ノード数項の重み')
    parser.add_argument('--proxy_lambda_single_child', default=1,     type=float,   help='単一子ノード項の重み')
    parser.add_argument('--proxy_round_tau', default=0.12, type=float, help='soft丸めの温度パラメータ')
    parser.add_argument('--proxy_mass_to_occ_gain', default=1.0, type=float, help='質量→占有変換のスケール')
    parser.add_argument('--octattention_teacher_device', default='auto', type=str, help='OctAttention teacherの実行先(auto/cuda/cpu/balanced)')
    parser.add_argument('--compression_rate_metric', default='total_bits', type=str, help='圧縮率損失の基準(total_bits/bits_per_point/bits_per_input_point)')

    """Test"""
    parser.add_argument('--input_dir_test', default=str(_data_subset_dir("ground")), type=str, help='テスト用入力点群のパス')
    parser.add_argument('--max_files_test', default=10, type=int, help='テスト時に読み込む最大ファイル数')
    parser.add_argument('--save_ply_dir', default=str((_LOG_ROOT / file_day / "MyNetwork_test" / "ply" / file_time).resolve()), type=str, help='出力点群の保存先')
    parser.add_argument('--test_compute_loss', default=True, type=str2bool, help='test.pyで幾何・圧縮統計をログ出力するか')
    parser.add_argument('--test_drop_threshold', default=0.50, type=float, help='test.pyで点削除ゲートをhard化するしきい値。全点keep/全点dropになる場合はsum(final_w)ベースのexpected_keepへ自動フォールバック')
    parser.add_argument('--test_adjust_threshold', default=1e-6, type=float, help='test.pyで点が調整されたと数える最小移動距離')
    parser.add_argument('--test_inference_mode', default='auto', type=str, help='推論方法(auto/full_cloud/subtree_merge/patch/direct/legacy)')
    parser.add_argument('--test_auto_time_tolerance', default=0.10, type=float, help='auto選択で時間差がこの比率以内ならメモリ節約側を優先')
    parser.add_argument('--test_subtree_level', default=0, type=int, help='subtree_merge時に使うSubtree深さ(0ならtrain_subtree_level/repair_unit_level)')
    parser.add_argument('--test_subtree_min_points', default=4, type=int, help='subtree_merge時に各Subtreeへ最低限含めたい点数')
    parser.add_argument('--test_metric_max_points', default=8192, type=int, help='testログ用CD/D1/D2計算で使う最大点数（0で全点）')
    parser.add_argument('--test_metric_normal_k', default=16, type=int, help='testログ用D2PSNRの法線推定k近傍数')

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
    parser.add_argument('--train_subtree_level', default=0, type=int, help='train subtree深さ(0ならrepair_unit_levelを使う)')
    parser.add_argument('--train_subtree_randomize_level', default=True, type=str2bool, help='train時にsubtree深さを一定範囲でランダム化するか')
    parser.add_argument('--train_subtree_level_jitter', default=1, type=int, help='train_subtree_levelの前後に何段までランダム化を許すか')
    parser.add_argument('--train_subtree_level_min', default=0, type=int, help='train時subtree深さの最小値(0ならbase-jitter)')
    parser.add_argument('--train_subtree_level_max', default=0, type=int, help='train時subtree深さの最大値(0ならbase+jitter)')
    parser.add_argument('--train_subtree_random_full_range', default=True, type=str2bool, help='min/max未指定時はデータから推定した全Octree深さ範囲からランダムに選ぶ')
    parser.add_argument('--train_subtree_level_sampling', default='uniform_random', type=str, help='subtree深さサンプリング方法(uniform_random/coverage_cycle)')
    parser.add_argument('--train_subtree_min_points', default=4, type=int, help='train時に優先的に選ぶsubtreeの最小点数（満たす候補が無ければフォールバック）')
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
    parser.add_argument('--plot_outlier_abs_threshold', default=1e10, type=float, help='この絶対値を超えるstep値をプロットから除外する閾値（0以下で無効）')
    parser.add_argument('--plot_outlier_rel_factor', default=1e4, type=float, help='直近履歴の中央値に対する倍率閾値（0以下で無効）')
    parser.add_argument('--plot_outlier_min_history', default=8, type=int, help='相対外れ値判定を始めるまでに必要な履歴数')
    parser.add_argument('--plot_outlier_history_window', default=64, type=int, help='相対外れ値判定で参照する直近履歴数（0で全履歴）')
    parser.add_argument('--plot_outlier_min_scale', default=1.0, type=float, help='相対外れ値判定で使う中央値の下限値')
    parser.add_argument('--retain_debug_tensors', default=False, type=str2bool, help='中間勾配を保持するか')
    parser.add_argument('--debug_grad_flow', default=False, type=str2bool, help='勾配ノルムをログ出力するか')
    parser.add_argument('--debug_grad_flow_rate', default=1, type=int, help='勾配ログの出力間隔')
    parser.add_argument('--train_grad_clip', default=0.0, type=float, help='学習時の勾配クリップ値（0で無効）')
    parser.add_argument('--debug_timing', default=False, type=str2bool, help='ステップ内の時間内訳をログ出力するか')
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
    if not _cli_option_was_provided("--input_dir_test"):
        args.input_dir_test = str(_data_subset_dir("ground", args.dataname, args.dataset_name))
    if not _cli_option_was_provided("--save_ply_dir"):
        args.save_ply_dir = str((_LOG_ROOT / args.date / "MyNetwork_test" / "ply" / args.time).resolve())

    args.octree_ctx_dim = max(int(args.octree_ctx_dim), 1)
    args.w_attr = float(args.w_prun)
    args.w_policy = float(args.w_add)
    args.w_repair = float(args.w_dis)

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
    if not _cli_option_was_provided("--out_path"):
        args.out_path = str((_LOG_ROOT / args.date / "MyNetwork_train" / "checkpoints" / args.run_name).resolve())
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
    }
    if args.compression_loss_backend not in valid_backends:
        raise ValueError(
            "--compression_loss_backend must be one of: proxy, octattention_actual, "
            "octattention_actual_ste, octattention_surrogate, sparsepcgc_actual, "
            "sparsepcgc_actual_ste, sparsepcgc_surrogate, gpcc_actual, "
            f"gpcc_actual_ste, gpcc_surrogate (got {args.compression_loss_backend})"
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
    args.compression_surrogate_forward_mode = str(
        getattr(args, "compression_surrogate_forward_mode", "surrogate")
    ).strip().lower()
    if args.compression_surrogate_forward_mode not in {"surrogate", "teacher_ste"}:
        raise ValueError("--compression_surrogate_forward_mode must be surrogate or teacher_ste")
    args.compression_surrogate_aux_node_weight = max(
        float(getattr(args, "compression_surrogate_aux_node_weight", 0.0)),
        0.0,
    )
    args.compression_surrogate_aux_single_weight = max(
        float(getattr(args, "compression_surrogate_aux_single_weight", 0.0)),
        0.0,
    )
    args.compression_octree_stat_depth = max(int(getattr(args, "compression_octree_stat_depth", 0)), 0)
    args.compression_octree_stat_force = bool(getattr(args, "compression_octree_stat_force", True))

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
    args.test_subtree_level = int(getattr(args, "test_subtree_level", 0))
    if args.test_subtree_level < 0:
        raise ValueError("--test_subtree_level must be >= 0")
    args.test_subtree_min_points = max(
        int(getattr(args, "test_subtree_min_points", getattr(args, "train_subtree_min_points", 1))),
        1,
    )
    args.test_metric_max_points = max(int(getattr(args, "test_metric_max_points", 8192)), 0)
    args.test_metric_normal_k = max(int(getattr(args, "test_metric_normal_k", 16)), 3)

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
    args.train_subtree_random_full_range = bool(getattr(args, "train_subtree_random_full_range", True))
    args.train_subtree_min_points = max(int(getattr(args, "train_subtree_min_points", 1)), 1)
    args.train_subtree_stat_log_limit = max(int(getattr(args, "train_subtree_stat_log_limit", 16)), 0)
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
    args.save_dir = _resolve_repo_or_cwd_path(args.save_dir)
    args.out_path = _resolve_repo_or_cwd_path(args.out_path)
    args.log_root = _resolve_repo_or_cwd_path(args.log_root)
    args.save_ply_dir = _resolve_repo_or_cwd_path(args.save_ply_dir)

    return args
