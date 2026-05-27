import torch
import torch.nn as nn
import time

from .encoder.point_trans import PointTransformer
from .utils.pointcloud import utils_repkpu
from .utils.pointcloud.utils_repkpu import get_knn_pts, index_points
from .utils.pointcloud.octree_subtree import assign_octree_subtree_keys, subtree_membership_mask
from .utils.pointcloud.sparse_tensor import build_sparse_point_tensor_single
from .modules.cause_aggregation import CauseDiagnosisAggregation
from .modules.cost_attribution import CAUSE_NAMES, CostAttributionModule
from .modules.octree_structure import OctreeStructureAnalysis
from .modules.structure_actuator import StructureRepairActuator
from .modules.structure_policy import POLICY_NAMES, StructureRepairPolicy


class Network(nn.Module):
    """
    ◆全体フロー
    ・入力点群
    ・PointTransformer：局所・大域特徴抽出
    ・OctreeStructureAnalisis：Octree構造特徴を計算
    ・CostAttributionModule：圧縮非効率の原因を推定
    ・CauseDiagnosisAggregation：原因スコアをSubtree/Repai Unit単位に集約
    ・StructureRepairPolicy：修復方策を選択
    ・StructureRepairActuator：点操作を実行
    """

    def __init__(self, args, writer):
        """基本情報セットアップ"""
        super().__init__() # 親クラスの初期化
        self.args = args
        self.writer = writer
        self.args.encoder_input_dim = 1 if bool(getattr(self.args, "encoder_sparse_tensor", True)) else 3 # Encoderに入力する特徴次元を決めている。encoder_sparse_tensor=TrueならSparse Tensor用に1次元特徴
        self.cache_enabled = False # 入力キャッシュの初期化
        self.input_cache = {} # 入力キャッシュ用の辞書
        self.expected_input_cache_entries = 0 # 想定されるキャッシュ数を初期化
        self.debug_tensors = {} # デバッグ用のテンソルを保存
        self.last_structure_debug = {} # 直近Forward時の構造診断デバッグ情報を保存する辞書の初期化
        self.last_encoder_debug = {} # 直近Forward時のEncoder関連デバッグ情報を保存する辞書を初期化
        self.last_runtime_timing = {} # 直近Forward時の実行時間計測結果を保存する辞書を初期化
        # self._configure_non_encoder_batchnorm() # Encoder以外の構造系モジュールに含まれるBatchNormの設定を変更する
        
        """モジュールセットアップ"""
        self.encoder = PointTransformer(self.args) # 特徴抽出器
        self.structure_analyzer = OctreeStructureAnalysis(self.args, self.writer) # Octree構造解析モジュールの作成

        """次元数セットアップ"""
        fused_dim = int(getattr(self.args, "fused_feat_dim", getattr(self.args, "out_dim", 64))) # Encoderが出す統合特徴の次元数
        structure_dim = int(self.structure_analyzer.feature_dim) # Octree構造解析モジュールが出す構造特徴の次元数
        hidden_dim = int(getattr(self.args, "structure_hidden_dim", 96)) # 構造診断・方策選択モジュール内部の隠れ層次元

        """モジュールセットアップ"""
        self.cost_attributor = CostAttributionModule(in_channels=fused_dim + structure_dim, hidden_dim=hidden_dim) # 圧縮非効率の現認を推定するモジュールの作成
        self.cause_aggregator = CauseDiagnosisAggregation(self.args) # 点単位・ノード単位の原因スコアを、SubtreeやRepai Unitに集約するモジュール
        self.policy_module = StructureRepairPolicy( # 構造修復方策を選ぶモジュール
            in_channels=fused_dim + structure_dim + len(CAUSE_NAMES) + self.cause_aggregator.priority_dim,
            hidden_dim=hidden_dim,
            temperature=float(getattr(self.args, "repair_policy_temperature", 1.0)),
        )
        self.actuator = StructureRepairActuator( # 実際に点群に操作するモジュールの作成
            in_channels=structure_dim + len(CAUSE_NAMES) + len(POLICY_NAMES) + self.cause_aggregator.priority_dim,
            hidden_dim=int(getattr(self.args, "repair_actuator_hidden_dim", 64)),
            args=self.args,
        )

        """旧モジュールセットアップ"""
        # self.prun_module = self.cost_attributor
        # self.adding_module = self.policy_module
        # self.disp_module = self.actuator

    def _configure_non_encoder_batchnorm(self): # Encoder以外のBatchNorm設定を調整するための補助関数
        if bool(getattr(self.args, "module_bn_use_running_stats", False)):
            return
        changed = 0
        for root in (self.cost_attributor, self.policy_module, self.actuator):
            for module in root.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                    module.num_batches_tracked = None
                    changed += 1
        if changed > 0 and self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(f"Structure modules use per-batch BatchNorm stats ({changed} layers).")

    def train(self, mode: bool = True): # Encoderを訓練対象から外すためにPyTorchのtrain()を上書き
        super().train(mode)
        if bool(getattr(self.args, "encoder_0grad", True)):
            self.encoder.eval()
        return self

    """キャッシュ関係の互換関数"""
    def input_cache_stats(self): # 入力キャッシュの状態を返す関数
        return {"entries": 0, "bytes": 0, "max_entries": 0, "max_bytes": 0}

    def set_expected_input_cache_entries(self, total_entries): # 想定されるキャッシュ件数を設定する関数
        self.expected_input_cache_entries = max(int(total_entries), 0)

    def estimate_input_cache_capacity_entries(self): # 入力キャッシュに何件保存できるか見積もる関数
        return 0

    def clear_input_cache(self): # 入力キャッシュを空にする関数
        self.input_cache.clear()

    def disable_input_cache(self): # 入力キャッシュを無効化する関数
        self.cache_enabled = False
        self.clear_input_cache()

    # def warmup_frozen_input(self, pts_xyz, cache_key=None, coord_scale=None): # 旧実装で、固定入力特徴を事前計算してキャッシュするための関数
    #     return None

    # def clear_discrete_policy_terms(self): # 離散方策に関する一時的な損失項やログ情報を消すための関数
    #     return None

    """補助関数"""
    def discrete_policy_loss(self, reward): # 離散方策に対する方策勾配風の損失を返すための関数
        return reward.new_zeros(())

    def _normalize_coord_scale(self, pts_xyz, coord_scale): # 座標スケールと点群テンソルに合わせた形に整える補助関数
        if coord_scale is None:
            return pts_xyz.new_ones((pts_xyz.shape[0], 1, 1))
        if torch.is_tensor(coord_scale):
            scale = coord_scale.to(device=pts_xyz.device, dtype=pts_xyz.dtype).reshape(-1, 1, 1)
            if scale.shape[0] == 1 and pts_xyz.shape[0] > 1:
                scale = scale.expand(pts_xyz.shape[0], -1, -1)
            return scale.clamp_min(1e-9)
        return pts_xyz.new_full((pts_xyz.shape[0], 1, 1), max(float(coord_scale), 1e-9))

    def _should_collect_runtime_debug(self): # このStepで構造デバッグ情報を集めるか否かの判定
        return bool(
            (getattr(self.args, "verbose_step_logs", False) and getattr(self.args, "_log_this_step", True))
            or getattr(self.args, "_collect_structure_debug", False)
        )

    def _timing_enabled(self): # Forward内で処理時間を測定するか否かの判定
        return bool(getattr(self.args, "debug_timing", False))

    @staticmethod
    def _sync_if_cuda_tensor(tensor): # 入力テンソルがCUDA上にある場合だけ、GPU処理を同期する関数
        if torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    @staticmethod
    def _first_occurrence_indices(inverse, num_unique): # Unique Voxelに対して、最初に出現した点のインデックスを1つずつ取り出す
        order = torch.argsort(inverse, stable=True)
        sorted_inverse = inverse.index_select(0, order)
        first_mask = torch.ones_like(sorted_inverse, dtype=torch.bool)
        if sorted_inverse.numel() > 1:
            first_mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        first_sorted_idx = order[first_mask]
        if int(first_sorted_idx.numel()) != int(num_unique):
            raise RuntimeError("Failed to recover one representative index per voxel.")
        return first_sorted_idx

    """Encoder用関数"""
    def _voxel_downsample_single(self, pts_xyz, coord_scale): # 1つの点群サンプルに対して、Voxel DownSamplingを行う関数
        num_points = int(pts_xyz.shape[-1]) # 点数取得
        max_points = max(int(getattr(self.args, "encoder_pre_downsample_max_points", 0)), 0) # Encoderに入力数最大点数
        cdist_cap = max(int(getattr(self.args, "encoder_cdist_max_points", 0)), 0) # 距離計算を使う場合の最大点数制限を取得
        if utils_repkpu.KNN_BACKEND != "pointops_cuda" and cdist_cap > 0:
            max_points = min(max_points, cdist_cap) if max_points > 0 else cdist_cap
        if max_points <= 0 or num_points <= max_points:
            full_idx = torch.arange(num_points, device=pts_xyz.device, dtype=torch.long)
            return pts_xyz, full_idx, 0.0

        scale = self._normalize_coord_scale(pts_xyz.unsqueeze(0), coord_scale).reshape(-1)[0] # 座標スケールの正規化
        base_voxel = ( # 規準となるVoxelサイズの計算
            float(getattr(self.args, "encoder_pre_downsample_voxel_scale", 1.0))
            * float(getattr(self.args, "qs", 2.0))
            / max(float(scale.item()), 1e-9)
        )
        mins = pts_xyz.amin(dim=1, keepdim=True) # 各座標軸ごとの最小値
        span = (pts_xyz.amax(dim=1, keepdim=True) - mins).amax().clamp_min(1e-9) # 点群全体の空間的な広がりの算出
        voxel_size = max(base_voxel, float(span.item()) / max(float(max_points), 1.0) ** (1.0 / 3.0)) # 実際に使うVoxelサイズの決定
        growth = max(float(getattr(self.args, "encoder_pre_downsample_growth", 1.5)), 1.05) # Voxelサイズを大きくしていく倍率の取得
        max_iters = max(int(getattr(self.args, "encoder_pre_downsample_max_iters", 8)), 1) # Voxelサイズ調整を試す最大反復回数の取得

        rep_idx = None # 各Voxelの代表点インデックスを保存する変数の初期化
        for _ in range(max_iters): # Voxelサイズを調整しながらDownSamplingを行う
            coords = torch.floor((pts_xyz - mins) / max(voxel_size, 1e-9)).long().transpose(0, 1).contiguous()
            unique_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            voxel_count = int(unique_coords.shape[0])
            rep_idx = self._first_occurrence_indices(inverse, voxel_count)
            if voxel_count <= max_points:
                break
            voxel_size *= growth

        rep_idx = torch.sort(rep_idx.unique(sorted=True)).values
        if rep_idx.numel() > max_points:
            rep_idx = rep_idx[:max_points]
        coarse_xyz = pts_xyz.index_select(1, rep_idx)
        return coarse_xyz, rep_idx, voxel_size

    def _propagate_encoder_features(self, full_xyz, coarse_xyz, coarse_feat): # 粗い点群上で得た特徴を元の点群の各点へ伝播する関数
        if coarse_xyz.shape[-1] == full_xyz.shape[-1]:
            return coarse_feat
        method = str(getattr(self.args, "encoder_feature_propagation", "knn_inverse_distance")).strip().lower()
        if method not in {"knn_inverse_distance", "idw", "nearest"}:
            raise ValueError(f"Unsupported encoder_feature_propagation: {method}")

        k = max(int(getattr(self.args, "encoder_feature_propagation_k", 3)), 1)
        k = min(k, int(coarse_xyz.shape[-1]))
        coarse_knn, knn_idx = get_knn_pts(k, coarse_xyz, full_xyz, return_idx=True)
        feat_knn = index_points(coarse_feat, knn_idx)

        if k == 1 or method == "nearest":
            return feat_knn[..., 0]

        dist = torch.linalg.norm(full_xyz.unsqueeze(-1) - coarse_knn, dim=1).clamp_min(1e-9)
        inv_dist = 1.0 / dist
        weight = inv_dist / inv_dist.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (feat_knn * weight.unsqueeze(1)).sum(dim=-1)

    def _propagate_batch_features(self, full_xyz, coarse_xyz, coarse_counts, coarse_feat): # バッチ単位で、粗い点群特徴を元点群へ伝播する関数
        if coarse_xyz.shape[-1] == full_xyz.shape[-1]:
            return coarse_feat
        propagated = []
        for b in range(full_xyz.shape[0]):
            coarse_count = coarse_counts[b]
            coarse_xyz_b = coarse_xyz[b:b + 1, :, :coarse_count]
            coarse_feat_b = coarse_feat[b:b + 1, :, :coarse_count]
            full_xyz_b = full_xyz[b:b + 1]
            propagated.append(self._propagate_encoder_features(full_xyz_b, coarse_xyz_b, coarse_feat_b))
        return torch.cat(propagated, dim=0)

    @staticmethod
    def _propagate_batch_features_by_index(coarse_feat, coarse_counts, full_to_coarse_idx): # 粗い点群上の特徴を元点分側へ戻す関数
        if not full_to_coarse_idx:
            return None
        propagated = []
        for b, idx in enumerate(full_to_coarse_idx):
            if idx is None:
                return None
            coarse_count = int(coarse_counts[b])
            if coarse_count <= 0:
                return None
            idx = idx.to(device=coarse_feat.device, dtype=torch.long).clamp(0, coarse_count - 1)
            propagated.append(coarse_feat[b:b + 1, :, :coarse_count].index_select(2, idx))
        return torch.cat(propagated, dim=0)

    def _encoder_sparse_qs_mode_and_pos(self): # Encoder用のSparse Tensorを作る時に、どの量子化スケール・量子化モード・位置量子化倍率を使うか決める関数
        compress_key = (
            str(getattr(self.args, "compress", ""))
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if compress_key == "sparsepcgc":
            voxel_size = max(float(getattr(self.args, "sparsepcgc_voxel_size", 1.0)), 1e-9)
            pos_q = max(int(getattr(self.args, "sparsepcgc_pos_quantscale", 1)), 1)
            return voxel_size, "sparsepcgc_twostep", pos_q
        if compress_key in {"gpcc", "gpcctmc3"}:
            return max(float(getattr(self.args, "gpcc_effective_qs", getattr(self.args, "qs", 2.0))), 1e-9), "floor_relative", 1
        return max(float(getattr(self.args, "qs", 2.0)), 1e-9), "floor_relative", 1

    def _encode(self, pts_xyz, coord_scale=None): # 入力点群をEncoderに通し、特徴量などを返す関数
        encoder_input = pts_xyz
        encoder_feat = None
        analysis_xyz = pts_xyz
        analysis_counts = []
        encoder_counts = []
        full_counts = []
        raw_counts = []
        pre_sparse_counts = []
        voxel_sizes = []
        full_to_coarse_idx = []
        if bool(getattr(self.args, "encoder_sparse_tensor", True)):
            sparse_xyz_list = []
            sparse_full_xyz_list = []
            sparse_feat_list = []
            scale = self._normalize_coord_scale(pts_xyz, coord_scale)
            sparse_qs, sparse_quant_mode, sparse_pos_q = self._encoder_sparse_qs_mode_and_pos()
            encoder_max_points = int(getattr(self.args, "encoder_pre_downsample_max_points", 0))
            cdist_cap = max(int(getattr(self.args, "encoder_cdist_max_points", 0)), 0)
            if utils_repkpu.KNN_BACKEND != "pointops_cuda" and cdist_cap > 0:
                encoder_max_points = min(encoder_max_points, cdist_cap) if encoder_max_points > 0 else cdist_cap
            for b in range(pts_xyz.shape[0]):
                sparse_tensor = build_sparse_point_tensor_single(
                    pts_xyz[b],
                    scale[b:b + 1],
                    max_points=encoder_max_points,
                    qs=sparse_qs,
                    raw_downsample_factor=float(getattr(self.args, "encoder_raw_downsample_factor", 1.0)),
                    voxel_scale=float(getattr(self.args, "encoder_pre_downsample_voxel_scale", 1.0)),
                    growth=float(getattr(self.args, "encoder_pre_downsample_growth", 1.5)),
                    max_iters=int(getattr(self.args, "encoder_pre_downsample_max_iters", 8)),
                    quant_mode=sparse_quant_mode,
                    pos_quantscale=sparse_pos_q,
                )
                sparse_full_xyz_b = sparse_tensor["sparse_xyz"]
                sparse_xyz_b = sparse_tensor["coords_xyz"]
                sparse_full_xyz_list.append(sparse_full_xyz_b)
                sparse_feat_b = sparse_tensor["feat"]
                sparse_xyz_list.append(sparse_xyz_b)
                sparse_feat_list.append(sparse_feat_b)
                raw_counts.append(int(sparse_tensor.get("raw_points", pts_xyz.shape[-1])))
                pre_sparse_counts.append(int(sparse_tensor.get("pre_downsample_points", pts_xyz.shape[-1])))
                analysis_counts.append(int(sparse_full_xyz_b.shape[-1]))
                encoder_counts.append(int(sparse_xyz_b.shape[-1]))
                full_counts.append(int(pts_xyz.shape[-1]))
                voxel_sizes.append(float(sparse_tensor["voxel_size"]))
                full_to_coarse_idx.append(sparse_tensor.get("full_to_coarse_idx"))
            if sparse_xyz_list:
                max_sparse = max(x.shape[-1] for x in sparse_xyz_list)
                padded_xyz = []
                padded_feat = []
                for sparse_xyz_b, sparse_feat_b in zip(sparse_xyz_list, sparse_feat_list):
                    if sparse_xyz_b.shape[-1] == max_sparse:
                        padded_xyz.append(sparse_xyz_b)
                        padded_feat.append(sparse_feat_b)
                        continue
                    pad_count = max_sparse - sparse_xyz_b.shape[-1]
                    pad_xyz = sparse_xyz_b[:, -1:].expand(-1, pad_count)
                    pad_feat = sparse_feat_b[:, -1:].expand(-1, pad_count)
                    padded_xyz.append(torch.cat([sparse_xyz_b, pad_xyz], dim=1))
                    padded_feat.append(torch.cat([sparse_feat_b, pad_feat], dim=1))
                encoder_input = torch.stack(padded_xyz, dim=0)
                encoder_feat = torch.stack(padded_feat, dim=0)
                analysis_xyz = torch.stack(sparse_full_xyz_list, dim=0)
        elif bool(getattr(self.args, "encoder_pre_downsample", False)):
            mode = str(getattr(self.args, "encoder_pre_downsample_mode", "voxel")).strip().lower()
            if mode != "voxel":
                raise ValueError(f"Unsupported encoder_pre_downsample_mode: {mode}")
            coarse_xyz_list = []
            scale = self._normalize_coord_scale(pts_xyz, coord_scale)
            for b in range(pts_xyz.shape[0]):
                coarse_xyz_b, rep_idx_b, voxel_size_b = self._voxel_downsample_single(
                    pts_xyz[b],
                    scale[b:b + 1],
                )
                coarse_xyz_list.append(coarse_xyz_b)
                encoder_counts.append(int(coarse_xyz_b.shape[-1]))
                analysis_counts.append(int(pts_xyz.shape[-1]))
                full_counts.append(int(pts_xyz.shape[-1]))
                voxel_sizes.append(float(voxel_size_b))
            if coarse_xyz_list:
                max_coarse = max(x.shape[-1] for x in coarse_xyz_list)
                if len(set(x.shape[-1] for x in coarse_xyz_list)) == 1:
                    encoder_input = torch.stack(coarse_xyz_list, dim=0)
                else:
                    padded = []
                    for coarse_xyz_b in coarse_xyz_list:
                        if coarse_xyz_b.shape[-1] == max_coarse:
                            padded.append(coarse_xyz_b)
                            continue
                        pad_count = max_coarse - coarse_xyz_b.shape[-1]
                        pad_xyz = coarse_xyz_b[:, -1:].expand(-1, pad_count)
                        padded.append(torch.cat([coarse_xyz_b, pad_xyz], dim=1))
                    encoder_input = torch.stack(padded, dim=0)
        else:
            analysis_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
            encoder_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
            full_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
            raw_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
            pre_sparse_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
            voxel_sizes = [0.0] * pts_xyz.shape[0]

        if bool(getattr(self.args, "encoder_0grad", True)):
            with torch.no_grad():
                local_sparse, fused_sparse = self.encoder(encoder_input, feat=encoder_feat)
            local_sparse = local_sparse.detach()
            fused_sparse = fused_sparse.detach()
        else:
            local_sparse, fused_sparse = self.encoder(encoder_input, feat=encoder_feat)

        local_feat = local_sparse
        fused_feat = fused_sparse
        keep_sparse_path = bool(
            getattr(self.args, "encoder_sparse_tensor", True)
            and getattr(self.args, "sparse_tensor_keep_after_encoder", True)
        )
        if keep_sparse_path:
            if encoder_input.shape[-1] != analysis_xyz.shape[-1]:
                local_by_index = self._propagate_batch_features_by_index(local_sparse, encoder_counts, full_to_coarse_idx)
                fused_by_index = self._propagate_batch_features_by_index(fused_sparse, encoder_counts, full_to_coarse_idx)
                if local_by_index is not None and fused_by_index is not None:
                    local_feat = local_by_index
                    fused_feat = fused_by_index
                else:
                    local_feat = self._propagate_batch_features(analysis_xyz, encoder_input, encoder_counts, local_sparse)
                    fused_feat = self._propagate_batch_features(analysis_xyz, encoder_input, encoder_counts, fused_sparse)
        elif encoder_input.shape[-1] != pts_xyz.shape[-1]:
            local_feat = self._propagate_batch_features(pts_xyz, encoder_input, encoder_counts, local_sparse)
            fused_feat = self._propagate_batch_features(pts_xyz, encoder_input, encoder_counts, fused_sparse)

        with torch.no_grad():
            if not analysis_counts:
                analysis_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
                encoder_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
                full_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
                raw_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
                pre_sparse_counts = [int(pts_xyz.shape[-1])] * pts_xyz.shape[0]
                voxel_sizes = [0.0] * pts_xyz.shape[0]
            self.last_encoder_debug = {
                "enabled": bool(getattr(self.args, "encoder_pre_downsample", False) or getattr(self.args, "encoder_sparse_tensor", True)),
                "mode": "sparse_tensor_voxel" if bool(getattr(self.args, "encoder_sparse_tensor", True)) else str(getattr(self.args, "encoder_pre_downsample_mode", "voxel")).strip().lower(),
                "raw_points": raw_counts,
                "pre_sparse_points": pre_sparse_counts,
                "analysis_points": analysis_counts,
                "full_points": full_counts,
                "coarse_points": encoder_counts,
                "voxel_size": voxel_sizes,
                "propagation": (
                    "voxel_assignment"
                    if keep_sparse_path and full_to_coarse_idx
                    else str(getattr(self.args, "encoder_feature_propagation", "knn_inverse_distance")).strip().lower()
                ),
                "propagation_k": int(getattr(self.args, "encoder_feature_propagation_k", 3)),
                "feature_dim": int(getattr(self.args, "encoder_input_dim", 3)),
                "kept_sparse_after_encoder": keep_sparse_path,
                "raw_downsample_factor": float(getattr(self.args, "encoder_raw_downsample_factor", 1.0)),
                "sparse_quant_mode": self._encoder_sparse_qs_mode_and_pos()[1],
            }
        return {
            "local_feat": local_feat,
            "fused_feat": fused_feat,
            "local_sparse": local_sparse,
            "fused_sparse": fused_sparse,
            "analysis_xyz": analysis_xyz,
            "encoder_xyz": encoder_input,
            "analysis_counts": analysis_counts,
            "coarse_counts": encoder_counts,
            "full_counts": full_counts,
            "voxel_sizes": voxel_sizes,
            "kept_sparse_after_encoder": keep_sparse_path,
        }

    """各モジュール用関数"""
    def _cause_weights(self, device, dtype): # 減員推定損失で各現員カテゴリの重みをTensorとして作る関数
        return torch.tensor(
            [
                float(getattr(self.args, "attr_node_weight", 1.0)),
                float(getattr(self.args, "attr_single_weight", 1.5)),
                float(getattr(self.args, "attr_lowprob_weight", 1.5)),
                float(getattr(self.args, "attr_context_weight", 1.0)),
                float(getattr(self.args, "attr_quant_weight", 1.25)),
                float(getattr(self.args, "attr_sparse_weight", 1.0)),
                float(getattr(self.args, "attr_outlier_weight", 1.0)),
                float(getattr(self.args, "attr_shape_weight", 0.75)),
            ],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _normalize_point_mask(mask, batch_size, num_points, device): # 点単位のマスクをBool Tensorに整形
        if mask is None:
            return None
        if mask.ndim == 3 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.ndim != 2 or mask.shape[0] != batch_size or mask.shape[1] != num_points:
            raise ValueError(f"point mask must have shape [B, N], got {tuple(mask.shape)}")
        return mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _masked_point_mean(values, point_mask): # 点毎の値について、マスク対象点だけの平均を算出
        if point_mask is None:
            return values.mean()
        if point_mask.ndim == 2:
            point_mask = point_mask.unsqueeze(1)
        mask = point_mask.to(device=values.device, dtype=values.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom

    @staticmethod
    def _argmax_count_dict(probs, names): # 確率Tensorから各点の最大確率カテゴリを選び、カテゴリ名ごとの出現回数を辞書で返す関数
        if probs is None or int(probs.numel()) <= 0:
            return {name: 0 for name in names}
        labels = probs.detach().argmax(dim=1).reshape(-1)
        return {
            name: int((labels == idx).sum().detach().cpu())
            for idx, name in enumerate(names)
        }

    @staticmethod
    def _operation_by_cause(cause_scores, policy_probs): # 各現員カテゴリごとに、その原因が最大になった点だけを集め、その点群に対する平均方策確率から代表的な修復操作を推定する関数
        if cause_scores is None or policy_probs is None or int(cause_scores.numel()) <= 0:
            return {}
        cause_labels = cause_scores.detach().argmax(dim=1)
        policy_det = policy_probs.detach()
        result = {}
        for cause_idx, cause_name in enumerate(CAUSE_NAMES):
            mask = cause_labels == cause_idx
            count = int(mask.sum().detach().cpu())
            if count <= 0:
                result[cause_name] = {
                    "count": 0,
                    "operation": "none",
                    "confidence": 0.0,
                }
                continue
            mask_f = mask.unsqueeze(1).to(device=policy_det.device, dtype=policy_det.dtype)
            policy_mean = (policy_det * mask_f).sum(dim=(0, 2)) / float(count)
            operation_idx = int(policy_mean.argmax().detach().cpu())
            result[cause_name] = {
                "count": count,
                "operation": POLICY_NAMES[operation_idx],
                "confidence": float(policy_mean[operation_idx].detach().cpu()),
            }
        return result

    def _patch_meta(self, pts_xyz, repair_gate, out_label): # 出力点群に対応するメタ情報を作る関数
        B, _, N = pts_xyz.shape
        anchor_idx = torch.arange(N, device=pts_xyz.device, dtype=torch.long).view(1, N).expand(B, N)
        valid_mask = torch.ones((B, N), device=pts_xyz.device, dtype=torch.bool)
        return {
            "anchor_idx_local": anchor_idx,
            "output_valid_mask": valid_mask,
            "out_label": out_label,
            "repair_gate": repair_gate.squeeze(1),
        }

    """原因推定/方策選択/編集制約損失"""
    def _build_losses(self, cause_scores, cause_logits, policy_logits, policy_teacher, cause_targets, edit_loss, point_mask=None): # 減員推定損失、方策選択損失、編集制約損失をそれぞれ計算する関数
        weights = self._cause_weights(cause_scores.device, cause_scores.dtype)
        attr_loss = self.cost_attributor.attribution_loss(
            cause_logits,
            cause_targets,
            weights=weights,
            point_mask=point_mask,
        )
        policy_loss = self.policy_module.policy_loss(
            policy_logits,
            policy_teacher,
            entropy_weight=float(getattr(self.args, "repair_policy_entropy_weight", 0.0)),
            point_mask=point_mask,
        )
        attr_loss = attr_loss * float(getattr(self.args, "loss_attr_scale", 1.0))
        policy_loss = policy_loss * float(getattr(self.args, "loss_policy_scale", 1.0))
        edit_loss = edit_loss * float(getattr(self.args, "loss_repair_scale", 1.0))
        return attr_loss, policy_loss, edit_loss

    """Network"""
    def forward( # Network全体の順伝播を行う関数
        self,
        pts_xyz,
        pts_attr,
        cache_key=None,
        return_patch_meta=False,
        coord_scale=None,
        return_attr_output=True,
        compute_internal_losses=None,
        subtree_ref=None,
        selected_subtree_keys=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        """セットアップ"""
        if pts_xyz.ndim != 3 or pts_xyz.shape[1] != 3: # 入力点群の形状チェック
            raise ValueError("pts_xyz must have shape [B, 3, N]")

        full_unit_keys = None # Subtree Key保存用の変数初期化
        selection_mask = None # 選択されたSubtreeに属する点だけを示すマスクの初期化
        if subtree_ref is not None: # Subtree参照情報が与えられているか確認
            full_unit_keys = assign_octree_subtree_keys(pts_xyz, subtree_ref)
            if selected_subtree_keys is not None:
                selected_subtree_keys = selected_subtree_keys.to(device=pts_xyz.device, dtype=full_unit_keys.dtype).reshape(-1)
                selection_mask = subtree_membership_mask(full_unit_keys, selected_subtree_keys)
                selection_mask = self._normalize_point_mask(
                    selection_mask,
                    batch_size=pts_xyz.shape[0],
                    num_points=pts_xyz.shape[2],
                    device=pts_xyz.device,
                )

        timing_enabled = self._timing_enabled() # 時間計測を行うか否か取得
        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_t0 = time.time()
            
        """Encoder"""
        encode_state = self._encode(pts_xyz, coord_scale=coord_scale) # 入力点群を_encodeに渡して特徴抽出を行う
        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_encode_end = time.time()
        fused_feat = encode_state["fused_feat"] # 統合抽出
        analysis_xyz = encode_state["analysis_xyz"] # Octree構造解析に使う点群座標を取り出す
        analysis_counts = encode_state["analysis_counts"] # analysis_xyzの有効点数をバッチごとに取り出す
        keep_sparse_path = encode_state["kept_sparse_after_encoder"] # Encoder後もSparse Tensor側の点群を規準に処理するかどうかを取り出す
        analysis_unit_keys = assign_octree_subtree_keys(analysis_xyz, subtree_ref) if subtree_ref is not None else None # 解析用点群に対してSubtree Keyを割り当てる
        analysis_selection_mask = subtree_membership_mask(analysis_unit_keys, selected_subtree_keys) if analysis_unit_keys is not None and selected_subtree_keys is not None else None # 解析用点群に対して、選択されたSubtreeに属する点だけを示すマスクを作る

        if timing_enabled: # 時間計測が有効か否か
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_structure_start = time.time()
            runtime_diagnosis_total = 0.0
            runtime_attribution_total = 0.0
            runtime_decision_total = 0.0

        if keep_sparse_path: # Sparse Tensor側の点群を規準として構造解析を行うか否か
            """変数の初期化"""
            # self.writer.write(f"Keep Sparse Path")
            structure_feat_full_list = []
            subtree_scores_full_list = []
            policy_probs_full_list = []
            repair_priority_full_list = []
            snap_delta_full_list = []
            single_proxy_full_list = []
            node_proxy_full_list = []
            lowprob_proxy_full_list = []
            quant_proxy_full_list = []
            cause_scores_means = []
            subtree_scores_means = []
            policy_probs_means = []
            loss_attr_terms = []
            loss_policy_terms = []
            cause_scores = None
            subtree_scores = None
            policy_probs = None
            cause_logits = None
            policy_logits = None
            cause_targets = None
            structure = None
            
            for b in range(pts_xyz.shape[0]): # バッチ内の各点群サンプルを1つずつ処理
                analysis_count = analysis_counts[b] # バッチb番目の解析用点群を取得
                analysis_xyz_b = analysis_xyz[b:b + 1, :, :analysis_count] # 解析用点群だけを取り出す
                fused_feat_b = fused_feat[b:b + 1, :, :analysis_count] # Encoder統合特徴だけを取り出す
                coord_scale_b = None if coord_scale is None else coord_scale[b:b + 1] # 座標スケールだけを取り出す

                if timing_enabled:
                    self._sync_if_cuda_tensor(pts_xyz)
                    runtime_diag_start = time.time()

                """Octree構造解析器"""
                structure_b = self.structure_analyzer(
                    analysis_xyz_b,
                    coord_scale=coord_scale_b,
                    subtree_tree=subtree_tree if b == 0 else None,
                    full_octree_context=full_octree_context if b == 0 else None,
                    octree_input_mode=octree_input_mode,
                ) # 解析用点群に対して、Octree構造解析を行う
                if timing_enabled:
                    self._sync_if_cuda_tensor(pts_xyz)
                    runtime_diag_end = time.time()
                    runtime_diagnosis_total += runtime_diag_end - runtime_diag_start
                structure_feat_b = structure_b["features"].to(device=pts_xyz.device, dtype=fused_feat_b.dtype) # 構造解析結果から構造特徴を取り出す
                
                """圧縮非効率原因推定器"""
                cause_targets_b = structure_b["cause_targets"].to(device=pts_xyz.device, dtype=fused_feat_b.dtype) # 構造解析結果から原因教師を取り出し、DeviceとDtypeを合わせることで、原因推定損失の教師信号として扱う
                cause_input_b = torch.cat([fused_feat_b, structure_feat_b], dim=1) # Encoder統合特徴と構造特徴をチャネル方向に結合
                if timing_enabled:
                    self._sync_if_cuda_tensor(pts_xyz)
                    runtime_attr_start = time.time()
                cause_scores_b, cause_logits_b = self.cost_attributor(cause_input_b) # 各点の圧縮非効率原因スコアとlogitsを推定
                if timing_enabled:
                    self._sync_if_cuda_tensor(pts_xyz)
                    runtime_attr_end = time.time()
                    runtime_attribution_total += runtime_attr_end - runtime_attr_start
                    runtime_decision_start = time.time()
                    
                """原因スコア集約器"""
                aggregated_b = self.cause_aggregator( # 原因スコアを点単位からSubtree/Repair Unit単位へ集約する
                    pts_xyz=analysis_xyz_b,
                    cause_scores=cause_scores_b,
                    cause_targets=cause_targets_b,
                    unit_keys=None if analysis_unit_keys is None else analysis_unit_keys[b:b + 1, :analysis_count],
                )
                subtree_scores_b = aggregated_b["scores"] # Subtree原因スコア
                subtree_targets_b = aggregated_b["targets"] # Subtree教師
                repair_priority_b = aggregated_b["priority"].to(device=pts_xyz.device, dtype=fused_feat_b.dtype) # DeviceとDtypeを合わせる
                
                """方策選択器"""
                policy_input_b = torch.cat([fused_feat_b, structure_feat_b, subtree_scores_b, repair_priority_b], dim=1) # 統合特徴、構造特徴、原因スコア、修復優先度をチャネル方向に結合
                policy_probs_b, policy_logits_b = self.policy_module(policy_input_b) # 各点又は各Repai Unitに対して修復操作の確率をLogitsを出す
                if timing_enabled:
                    self._sync_if_cuda_tensor(pts_xyz)
                    runtime_decision_end = time.time()
                    runtime_decision_total += runtime_decision_end - runtime_decision_start

                """点操作前処理"""
                full_xyz_b = pts_xyz[b:b + 1]
                if analysis_xyz_b.shape[-1] == full_xyz_b.shape[-1]:
                    structure_feat_full_list.append(structure_feat_b)
                    subtree_scores_full_list.append(subtree_scores_b)
                    policy_probs_full_list.append(policy_probs_b)
                    repair_priority_full_list.append(repair_priority_b)
                    snap_delta_full_list.append(structure_b["snap_delta"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                    single_proxy_full_list.append(structure_b["single_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                    node_proxy_full_list.append(structure_b["node_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                    lowprob_proxy_full_list.append(structure_b["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                    quant_proxy_full_list.append(structure_b["quant_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                else:
                    structure_feat_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, structure_feat_b))
                    subtree_scores_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, subtree_scores_b))
                    policy_probs_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, policy_probs_b))
                    repair_priority_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, repair_priority_b))
                    snap_delta_b = structure_b["snap_delta"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    snap_delta_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, snap_delta_b))
                    single_proxy_b = structure_b["single_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    node_proxy_b = structure_b["node_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    lowprob_proxy_b = structure_b["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    quant_proxy_b = structure_b["quant_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    single_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, single_proxy_b))
                    node_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, node_proxy_b))
                    lowprob_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, lowprob_proxy_b))
                    quant_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, quant_proxy_b))

                """平均値の保存"""
                cause_scores_means.append(cause_scores_b.mean(dim=2))
                subtree_scores_means.append(subtree_scores_b.mean(dim=2))
                policy_probs_means.append(policy_probs_b.mean(dim=2))
                
                """raw auxiliary internal lossの計算"""
                if compute_internal_losses is None:
                    compute_losses_b = self.training
                else:
                    compute_losses_b = compute_internal_losses
                if compute_losses_b:
                    analysis_mask_b = None
                    if analysis_selection_mask is not None:
                        analysis_mask_b = analysis_selection_mask[b:b + 1, :analysis_count]
                    policy_teacher_b = self.policy_module.build_teacher(subtree_targets_b)
                    loss_attr_terms.append(self.cost_attributor.attribution_loss(
                        cause_logits_b,
                        cause_targets_b,
                        weights=self._cause_weights(cause_scores_b.device, cause_scores_b.dtype),
                        point_mask=analysis_mask_b,
                    ))
                    loss_policy_terms.append(self.policy_module.policy_loss(
                        policy_logits_b,
                        policy_teacher_b,
                        entropy_weight=float(getattr(self.args, "repair_policy_entropy_weight", 0.0)),
                        point_mask=analysis_mask_b,
                    ))

            """情報結合"""
            structure_feat_full = torch.cat(structure_feat_full_list, dim=0) # サンプルごとに保存していた全点群側の構造特徴をバッチ次元で結合
            subtree_scores_full = torch.cat(subtree_scores_full_list, dim=0) # サンプルごとに保存していた全点群側のSubtree原因スコアを、バッチ次元で結合
            policy_probs_full = torch.cat(policy_probs_full_list, dim=0) # サンプルごとに保存していた全点群側の修復方策確率を、バッチ次元で結合
            repair_priority_full = torch.cat(repair_priority_full_list, dim=0) # サンプルごとに保存していた全点群側の修復優先度を、バッチ次元で結合
            
            structure = { # Actuatorや診断値計算で使う構造情報の辞書設定
                "features": structure_feat_full,
                "snap_delta": torch.cat(snap_delta_full_list, dim=0),
                "single_proxy_full": torch.cat(single_proxy_full_list, dim=0),
                "node_proxy_full": torch.cat(node_proxy_full_list, dim=0),
                "lowprob_proxy_full": torch.cat(lowprob_proxy_full_list, dim=0),
                "quant_proxy_full": torch.cat(quant_proxy_full_list, dim=0),
                "level_debug": structure_b.get("level_debug") if structure_b is not None else None,
                "octree_input_mode": structure_b.get("octree_input_mode", "local_recomputed") if structure_b is not None else "local_recomputed",
                "octree_input_mode_requested": structure_b.get("octree_input_mode_requested", octree_input_mode) if structure_b is not None else octree_input_mode,
                "structural_voxel_mode": structure_b.get("structural_voxel_mode", "local_recomputed") if structure_b is not None else "local_recomputed",
                "point_feature_voxel_mode": structure_b.get("point_feature_voxel_mode", "local_xyz") if structure_b is not None else "local_xyz",
                "structural_voxel_key": structure_b.get("structural_voxel_key") if structure_b is not None else None,
                "point_feature_voxel_key": structure_b.get("point_feature_voxel_key") if structure_b is not None else None,
            }

            cause_mean = torch.stack(cause_scores_means, dim=0).mean(dim=0).squeeze(0).detach().cpu()
            subtree_mean = torch.stack(subtree_scores_means, dim=0).mean(dim=0).squeeze(0).detach().cpu()
            policy_mean = torch.stack(policy_probs_means, dim=0).mean(dim=0).squeeze(0).detach().cpu()
            if loss_attr_terms:
                loss_attr_sparse = torch.stack(loss_attr_terms).mean()
                loss_policy_sparse = torch.stack(loss_policy_terms).mean()
            else:
                loss_attr_sparse = pts_xyz.new_zeros(())
                loss_policy_sparse = pts_xyz.new_zeros(())
        else:
            analysis_xyz = pts_xyz
            if timing_enabled:
                self._sync_if_cuda_tensor(pts_xyz)
                runtime_diag_start = time.time()
            structure = self.structure_analyzer(
                analysis_xyz,
                coord_scale=coord_scale,
                subtree_tree=subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
            if timing_enabled:
                self._sync_if_cuda_tensor(pts_xyz)
                runtime_diag_end = time.time()
                runtime_diagnosis_total += runtime_diag_end - runtime_diag_start
            structure_feat = structure["features"].to(device=pts_xyz.device, dtype=fused_feat.dtype)
            cause_targets = structure["cause_targets"].to(device=pts_xyz.device, dtype=fused_feat.dtype)

            cause_input = torch.cat([fused_feat, structure_feat], dim=1)
            if timing_enabled:
                self._sync_if_cuda_tensor(pts_xyz)
                runtime_attr_start = time.time()
            cause_scores, cause_logits = self.cost_attributor(cause_input)
            if timing_enabled:
                self._sync_if_cuda_tensor(pts_xyz)
                runtime_attr_end = time.time()
                runtime_attribution_total += runtime_attr_end - runtime_attr_start
                runtime_decision_start = time.time()
            aggregated = self.cause_aggregator(
                pts_xyz=analysis_xyz,
                cause_scores=cause_scores,
                cause_targets=cause_targets,
                unit_keys=full_unit_keys,
            )
            subtree_scores = aggregated["scores"]
            subtree_targets = aggregated["targets"]
            repair_priority = aggregated["priority"].to(device=pts_xyz.device, dtype=fused_feat.dtype)
            policy_input = torch.cat([fused_feat, structure_feat, subtree_scores, repair_priority], dim=1)
            policy_probs, policy_logits = self.policy_module(policy_input)
            if timing_enabled:
                self._sync_if_cuda_tensor(pts_xyz)
                runtime_decision_end = time.time()
                runtime_decision_total += runtime_decision_end - runtime_decision_start
            structure_feat_full = structure_feat
            subtree_scores_full = subtree_scores
            policy_probs_full = policy_probs
            repair_priority_full = repair_priority
            structure["single_proxy_full"] = structure["single_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            structure["node_proxy_full"] = structure["node_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            structure["lowprob_proxy_full"] = structure["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            structure["quant_proxy_full"] = structure["quant_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            loss_attr_sparse = None
            loss_policy_sparse = None

        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_structure_end = time.time()

        """点操作実行"""
        actuator_input = torch.cat([structure_feat_full, subtree_scores_full, policy_probs_full, repair_priority_full], dim=1) # 構造特徴、原因スコアなどをチャネル方向に結合
        pts_out, final_w, edit_loss, actuator_stats = self.actuator( # 実際に点操作を行う
            pts_xyz=pts_xyz,
            structure=structure,
            cause_scores=subtree_scores_full,
            policy_probs=policy_probs_full,
            actuator_features=actuator_input,
            repair_priority=repair_priority_full,
            coord_scale=coord_scale,
            selection_mask=selection_mask,
        )
        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_actuator_end = time.time()

        """scaled final internal lossの計算"""
        if compute_internal_losses is None:
            compute_internal_losses = self.training
        if compute_internal_losses:
            if keep_sparse_path:
                loss_attr = loss_attr_sparse * float(getattr(self.args, "loss_attr_scale", 1.0))
                loss_policy = loss_policy_sparse * float(getattr(self.args, "loss_policy_scale", 1.0))
                loss_repair = edit_loss * float(getattr(self.args, "loss_repair_scale", 1.0))
            else:
                policy_teacher = self.policy_module.build_teacher(subtree_targets)
                loss_attr, loss_policy, loss_repair = self._build_losses(
                    cause_scores=cause_scores,
                    cause_logits=cause_logits,
                    policy_logits=policy_logits,
                    policy_teacher=policy_teacher,
                    cause_targets=cause_targets,
                    edit_loss=edit_loss,
                    point_mask=selection_mask,
                )
        else:
            loss_attr = pts_xyz.new_zeros(())
            loss_policy = pts_xyz.new_zeros(())
            loss_repair = pts_xyz.new_zeros(())

        out_label = pts_xyz.new_zeros((pts_xyz.shape[0], pts_xyz.shape[2]))
        repair_gate = actuator_stats["repair_gate"]

        """問題スコアの算出"""
        single_chain_score = self._masked_point_mean(structure["single_proxy_full"].pow(2), selection_mask) # single_proxy_fullを二乗し、選択マスクがある場合はその範囲だけで平均して、単一子ノードの強さを表すスカラー値を算出
        lowprob_score = self._masked_point_mean(structure["lowprob_proxy_full"], selection_mask) # 低確率施入パターンのプロキシ値を、選択マスクがある場合はその範囲だけで平均して問題度を表すスカラー値にする
        lowprob_ratio = self._masked_point_mean( # 低確率Occupancyと判定される点の割合を計算する処理を開始
            (structure["lowprob_proxy_full"] > 0.5).to(dtype=pts_xyz.dtype),
            selection_mask,
        )
        node_score = self._masked_point_mean(structure["node_proxy_full"], selection_mask) # Octree Node数や局所Node構造の問題を表す変数で、Node系の構造問題スコアにする
        quant_score = self._masked_point_mean(structure["quant_proxy_full"], selection_mask) # 量子化由来の問題を表す変数で、量子化の構造問題スコアにする
        
        """ログ"""
        if self._should_collect_runtime_debug():
            with torch.no_grad():
                if not keep_sparse_path:
                    policy_mean = policy_probs.mean(dim=(0, 2)).detach().cpu()
                    cause_mean = cause_scores.mean(dim=(0, 2)).detach().cpu()
                    subtree_mean = subtree_scores.mean(dim=(0, 2)).detach().cpu()
                debug_cause_scores = subtree_scores_full.detach()
                debug_policy_probs = policy_probs_full.detach()
                policy_entropy = (
                    -(debug_policy_probs.clamp_min(1e-6).log() * debug_policy_probs).sum(dim=1).mean()
                )
                cause_argmax_counts = self._argmax_count_dict(debug_cause_scores, CAUSE_NAMES)
                policy_argmax_counts = self._argmax_count_dict(debug_policy_probs, POLICY_NAMES)
                active_policy_count = sum(1 for value in policy_argmax_counts.values() if value > 0)
                self.last_structure_debug = {
                    "loss_attr": float(loss_attr.detach().cpu()),
                    "loss_policy": float(loss_policy.detach().cpu()),
                    "loss_repair": float(loss_repair.detach().cpu()),
                        "repair_ratio": float(repair_gate.mean().detach().cpu()),
                        "add_ratio": float(actuator_stats.get("add_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                        "add_prob_mean": float(actuator_stats.get("add_prob_mean", pts_xyz.new_zeros(())).detach().cpu()),
                        "add_prob_max": float(actuator_stats.get("add_prob_max", pts_xyz.new_zeros(())).detach().cpu()),
                        "add_priority_mean": float(actuator_stats.get("add_priority_mean", pts_xyz.new_zeros(())).detach().cpu()),
                        "add_priority_max": float(actuator_stats.get("add_priority_max", pts_xyz.new_zeros(())).detach().cpu()),
                        "add_count": int(actuator_stats.get("add_count", 0)),
                    "add_effective_count": int(actuator_stats.get("add_effective_count", 0)),
                    "add_candidate_ratio": float(actuator_stats.get("add_candidate_ratio", 0.0)),
                    # Prune/Add/Adjustの学習済み実行量をログへ渡し、固定比率への張り付きを確認できるようにする。
                    "learned_drop_ratio": float(actuator_stats.get("learned_drop_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                    "learned_drop_ratio_std": float(actuator_stats.get("learned_drop_ratio_std", pts_xyz.new_zeros(())).detach().cpu()),
                    "learned_add_ratio": float(actuator_stats.get("learned_add_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                    "learned_add_ratio_std": float(actuator_stats.get("learned_add_ratio_std", pts_xyz.new_zeros(())).detach().cpu()),
                    "learned_move_ratio": float(actuator_stats.get("learned_move_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                    "learned_move_ratio_std": float(actuator_stats.get("learned_move_ratio_std", pts_xyz.new_zeros(())).detach().cpu()),
                    "operation_amount_consistency_loss": float(actuator_stats.get("operation_amount_consistency_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "operation_entropy": float(actuator_stats.get("operation_entropy", pts_xyz.new_zeros(())).detach().cpu()),
                    "operation_prob_floor_applied": bool(actuator_stats.get("operation_prob_floor_applied", False)),
                    "temperature": float(actuator_stats.get("temperature", 0.0)),
                    "exploration_noise": float(actuator_stats.get("exploration_noise", 0.0)),
                    "sparsepcgc_add_experiment_enabled": bool(actuator_stats.get("sparsepcgc_add_experiment_enabled", False)),
                    "sparsepcgc_add_warmup": float(actuator_stats.get("sparsepcgc_add_warmup", 1.0)),
                        "add_score_noise": float(actuator_stats.get("add_score_noise", 0.0)),
                        "add_weight_random_mix": float(actuator_stats.get("add_weight_random_mix", 0.0)),
                        "drop_score_noise": float(actuator_stats.get("drop_score_noise", 0.0)),
                        # Adjust score探索ノイズ量を記録し、初期探索が効いているか確認する。
                        "move_score_noise": float(actuator_stats.get("move_score_noise", 0.0)),
                        "drop_random_mix": float(actuator_stats.get("drop_random_mix", 0.0)),
                        "add_enabled": bool(actuator_stats.get("add_enabled", False)),
                        "prune_enabled": bool(actuator_stats.get("prune_enabled", False)),
                        "disp_enabled": bool(actuator_stats.get("disp_enabled", False)),
                        "actuator_stage": str(actuator_stats.get("actuator_stage", "unknown")),
                        "actuator_stage_raw": str(actuator_stats.get("actuator_stage_raw", "unknown")),
                        "actuator_strength": float(actuator_stats.get("actuator_strength", 0.0)),
                        "force_joint_actuator": bool(actuator_stats.get("force_joint_actuator", False)),
                        "add_drop_conflict_loss": float(actuator_stats.get("add_drop_conflict_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "added_keep_loss": float(actuator_stats.get("added_keep_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "add_min_offset_loss": float(actuator_stats.get("add_min_offset_loss", pts_xyz.new_zeros(())).detach().cpu()),
                        "drop_ratio": float(actuator_stats["drop_prob"].mean().detach().cpu()),
                        "hard_drop_ratio": float(actuator_stats.get("hard_drop_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                        "hard_drop_count": int(actuator_stats.get("hard_drop_count", 0)),
                        "keep_ratio": float(actuator_stats["keep_prob"].mean().detach().cpu()),
                    "delta_norm": float(actuator_stats["delta"].norm(dim=1).mean().detach().cpu()),
                        "move_ratio": float(actuator_stats.get("move_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                        "hard_move_count": int(actuator_stats.get("hard_move_count", 0)),
                        "move_score_mean": float(actuator_stats.get("move_score_mean", pts_xyz.new_zeros(())).detach().cpu()),
                    "move_target_valid_ratio": float(actuator_stats.get("move_target_valid_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                    "moved_delta_mean": float(actuator_stats.get("moved_delta_mean", pts_xyz.new_zeros(())).detach().cpu()),
                    "before_occupied_voxel_count": int(actuator_stats.get("before_occupied_voxel_count", 0)),
                    "after_occupied_voxel_count": int(actuator_stats.get("after_occupied_voxel_count", 0)),
                    "occupied_voxel_delta": int(actuator_stats.get("occupied_voxel_delta", 0)),
                    "delete_target_voxel_count": int(actuator_stats.get("delete_target_voxel_count", 0)),
                    "delete_emptied_voxel_count": int(actuator_stats.get("delete_emptied_voxel_count", 0)),
                    "delete_removed_point_count": int(actuator_stats.get("delete_removed_point_count", 0)),
                    "add_target_voxel_count": int(actuator_stats.get("add_target_voxel_count", 0)),
                    "add_actual_point_count": int(actuator_stats.get("add_actual_point_count", 0)),
                    "move_source_voxel_count": int(actuator_stats.get("move_source_voxel_count", 0)),
                    "move_target_voxel_count": int(actuator_stats.get("move_target_voxel_count", 0)),
                    "adjusted_point_count": int(actuator_stats.get("adjusted_point_count", actuator_stats.get("hard_move_count", 0))),
                    "adjusted_point_rate": float(actuator_stats.get("adjusted_point_rate", pts_xyz.new_zeros(())).detach().cpu()) if torch.is_tensor(actuator_stats.get("adjusted_point_rate", None)) else float(actuator_stats.get("adjusted_point_rate", 0.0)),
                    "raw_hard_move_count_before_sparsepcgc_guard": int(actuator_stats.get("raw_hard_move_count_before_sparsepcgc_guard", actuator_stats.get("hard_move_count", 0))),
                    "source_unique_voxel_count": int(actuator_stats.get("source_unique_voxel_count", 0)),
                    "target_unique_voxel_count": int(actuator_stats.get("target_unique_voxel_count", 0)),
                    "target_duplicate_voxel_count": int(actuator_stats.get("target_duplicate_voxel_count", 0)),
                    "target_voxel_duplicate_rate": float(actuator_stats.get("target_voxel_duplicate_rate", pts_xyz.new_zeros(())).detach().cpu()) if torch.is_tensor(actuator_stats.get("target_voxel_duplicate_rate", None)) else float(actuator_stats.get("target_voxel_duplicate_rate", 0.0)),
                    "target_existing_occupied_count": int(actuator_stats.get("target_existing_occupied_count", 0)),
                    "target_existing_occupied_rate": float(actuator_stats.get("target_existing_occupied_rate", pts_xyz.new_zeros(())).detach().cpu()) if torch.is_tensor(actuator_stats.get("target_existing_occupied_rate", None)) else float(actuator_stats.get("target_existing_occupied_rate", 0.0)),
                    "target_empty_voxel_count": int(actuator_stats.get("target_empty_voxel_count", 0)),
                    "target_empty_voxel_rate": float(actuator_stats.get("target_empty_voxel_rate", pts_xyz.new_zeros(())).detach().cpu()) if torch.is_tensor(actuator_stats.get("target_empty_voxel_rate", None)) else float(actuator_stats.get("target_empty_voxel_rate", 0.0)),
                    "empty_target_violation_loss": float(actuator_stats.get("empty_target_violation_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "target_duplicate_voxel_loss": float(actuator_stats.get("target_duplicate_voxel_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "enable_sparsepcgc_empty_target_guard": bool(actuator_stats.get("enable_sparsepcgc_empty_target_guard", False)),
                    "enable_sparsepcgc_target_duplicate_guard": bool(actuator_stats.get("enable_sparsepcgc_target_duplicate_guard", False)),
                    "sparsepcgc_empty_target_guard_rejected_count": int(actuator_stats.get("sparsepcgc_empty_target_guard_rejected_count", 0)),
                    "sparsepcgc_target_duplicate_guard_rejected_count": int(actuator_stats.get("sparsepcgc_target_duplicate_guard_rejected_count", 0)),
                    "sparsepcgc_guard_rejected_count": int(actuator_stats.get("sparsepcgc_guard_rejected_count", 0)),
                    "sparsepcgc_move_existing_target_only": bool(actuator_stats.get("sparsepcgc_move_existing_target_only", False)),
                    "repair_move_require_empty_target": bool(actuator_stats.get("repair_move_require_empty_target", True)),
                    "repair_move_require_empty_target_effective": bool(actuator_stats.get("repair_move_require_empty_target_effective", True)),
                    "repair_move_max_points_per_voxel": int(actuator_stats.get("repair_move_max_points_per_voxel", 0)),
                    "max_move_ratio": float(actuator_stats.get("max_move_ratio", 0.0)),
                    "repair_move_hard_threshold": float(actuator_stats.get("repair_move_hard_threshold", 0.0)),
                    "moved_different_voxel_count": int(actuator_stats.get("moved_different_voxel_count", 0)),
                    "same_voxel_adjust_count": int(actuator_stats.get("same_voxel_adjust_count", 0)),
                    "preserve_ratio": float(actuator_stats.get("preserve_ratio", pts_xyz.new_zeros(())).detach().cpu()),
                    "quant_score": float(quant_score.detach().cpu()),
                    "quant_move_conflict_loss": float(actuator_stats.get("quant_move_conflict_loss", pts_xyz.new_zeros(())).detach().cpu()),
                    "local_edit_guard": float(actuator_stats.get("local_edit_guard", pts_xyz.new_zeros(())).detach().cpu()),
                    "analysis_points_mean": float(sum(analysis_counts) / max(len(analysis_counts), 1)) if keep_sparse_path else float(pts_xyz.shape[-1]),
                    "cause_mean": {
                        name: float(cause_mean[i])
                        for i, name in enumerate(CAUSE_NAMES)
                    },
                    "subtree_cause_mean": {
                        name: float(subtree_mean[i])
                        for i, name in enumerate(CAUSE_NAMES)
                    },
                    "policy_mean": {
                        name: float(policy_mean[i])
                        for i, name in enumerate(POLICY_NAMES)
                    },
                    "cause_argmax_counts": cause_argmax_counts,
                    "policy_argmax_counts": policy_argmax_counts,
                    "operation_by_cause": self._operation_by_cause(debug_cause_scores, debug_policy_probs),
                    "policy_entropy": float(policy_entropy.detach().cpu()),
                    "occupancy_nll_proxy": float(lowprob_score.detach().cpu()),
                    "lowprob_occupancy_ratio": float(lowprob_ratio.detach().cpu()),
                    "single_chain_score": float(single_chain_score.detach().cpu()),
                    "node_score": float(node_score.detach().cpu()),
                    "policy_diversity": int(active_policy_count),
                    "octree_level_debug": structure.get("level_debug"),
                    "use_subtree_tree": bool(subtree_tree is not None),
                    "use_full_octree_context": bool(full_octree_context is not None),
                    "octree_input_mode": str(structure.get("octree_input_mode", "local_recomputed")),
                    "octree_input_mode_requested": str(structure.get("octree_input_mode_requested", octree_input_mode)),
                    "structural_voxel_mode": str(structure.get("structural_voxel_mode", "local_recomputed")),
                    "point_feature_voxel_mode": str(structure.get("point_feature_voxel_mode", "local_xyz")),
                    "structural_voxel_key_available": bool(structure.get("structural_voxel_key") is not None),
                    "point_feature_voxel_key_available": bool(structure.get("point_feature_voxel_key") is not None),
                    "selected_subtree_key": str((subtree_tree or {}).get("subtree_key", "")),
                    "selected_subtree_path": str((subtree_tree or {}).get("subtree_path", "")),
                    "root_to_subtree_path": " > ".join((full_octree_context or {}).get("root_to_subtree_path", [])),
                    "global_offset": str((subtree_tree or full_octree_context or {}).get("global_offset", "")),
                    "local_offset": "encoder_or_point_feature_local_xyz",
                    "global_depth": int((subtree_tree or full_octree_context or {}).get("global_depth", 0) or 0),
                    "local_depth": int(getattr(self.structure_analyzer, "max_depth", 0)),
                    "parent_occupancy_code": int((full_octree_context or {}).get("parent_occupancy_code", 0) or 0),
                    "sibling_count": int(len((full_octree_context or {}).get("sibling_paths", []))),
                    "enable_sparsepcgc_exact_occupancy_teacher": bool(getattr(self.args, "enable_sparsepcgc_exact_occupancy_teacher", False)),
                    "sparsepcgc_exact_teacher_mode": str(getattr(self.args, "_current_exact_teacher_mode", getattr(self.args, "sparsepcgc_exact_teacher_mode", "auto"))),
                    "exact_teacher_uses_full_context": bool(getattr(self.args, "_current_exact_teacher_uses_full_context", False)),
                    "exact_teacher_fallback_reason": str(getattr(self.args, "_current_exact_teacher_fallback_reason", "")),
                }
        else:
            self.last_structure_debug = {}

        if timing_enabled:
            actuator_runtime = getattr(self.actuator, "last_runtime_timing", {}) or {}
            self.last_runtime_timing = {
                "encode": round(runtime_encode_end - runtime_t0, 6),
                "structure": round(runtime_structure_end - runtime_structure_start, 6),
                "actuator": round(runtime_actuator_end - runtime_structure_end, 6),
                "total_forward": round(runtime_actuator_end - runtime_t0, 6),
                "feature_extraction": round(runtime_encode_end - runtime_t0, 6),
                "structure_diagnosis": round(runtime_diagnosis_total, 6),
                "codec_cost_attribution": round(runtime_attribution_total, 6),
                "point_edit_decision": round(runtime_decision_total, 6),
                "delete_module": round(float(actuator_runtime.get("delete", 0.0)), 6),
                "add_module": round(float(actuator_runtime.get("add", 0.0)), 6),
                "adjust_move_module": round(float(actuator_runtime.get("adjust_move", 0.0)), 6),
                "postprocess": round(float(actuator_runtime.get("postprocess", 0.0)), 6),
                "actuator_setup": round(float(actuator_runtime.get("setup", 0.0)), 6),
            }
        else:
            self.last_runtime_timing = {}

        if return_attr_output and pts_attr is not None and pts_attr.shape[-1] == pts_out.shape[-1]:
            if pts_attr.shape[1] > 3:
                attr_channels = pts_attr[:, 3:, :]
            elif pts_attr.shape[1] > 0:
                attr_channels = pts_attr
            else:
                attr_channels = None
            pts_out_full = torch.cat([pts_out, attr_channels], dim=1) if attr_channels is not None else pts_out
        else:
            pts_out_full = pts_out

        output = (
            pts_out_full,
            loss_attr,
            loss_policy,
            loss_repair,
            final_w,
            single_chain_score,
            lowprob_score,
            node_score,
            out_label,
        )
        if return_patch_meta:
            return (*output, self._patch_meta(pts_xyz, repair_gate, out_label))
        return output
