import torch
import torch.nn as nn
import time

from .encoder.point_trans import PointTransformer
from .utils.pointcloud import utils_repkpu
from .utils.pointcloud.utils_repkpu import get_knn_pts, index_points
from .utils.pointcloud.octree_subtree import assign_octree_subtree_keys, subtree_membership_mask
from .utils.pointcloud.sparse_tensor import build_sparse_point_tensor_single
from .utils.pointcloud.sparsepcgc_voxel import (
    quantize_sparsepcgc_coords,
    unique_voxel_coords_batched,
    restore_points_from_voxel_coords,
)
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
        self.last_actuator_soft_terms = {} # 直近Forward時の微分可能な点操作proxyを保存する辞書
        self.last_runtime_timing = {} # 直近Forward時の実行時間計測結果を保存する辞書を初期化

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

    """Node/Voxel入力用関数"""
    @staticmethod
    def _normalize_node_voxel_coords(coords, device=None):
        if coords is None:
            return None
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)
        if device is not None:
            coords = coords.to(device=device)
        if coords.ndim == 2:
            if coords.shape[0] == 3:
                coords = coords.unsqueeze(0)
            elif coords.shape[1] == 3:
                coords = coords.transpose(0, 1).contiguous().unsqueeze(0)
            else:
                return None
        elif coords.ndim == 3:
            if coords.shape[1] == 3:
                coords = coords.contiguous()
            elif coords.shape[2] == 3:
                coords = coords.permute(0, 2, 1).contiguous()
            else:
                return None
        else:
            return None
        return coords.to(dtype=torch.long).contiguous()

    @staticmethod
    def _normalize_unit_keys(unit_keys, batch_size, point_count, device):
        """
        CauseDiagnosisAggregationへ渡すunit_keysを [B, N] に揃える。
        global_morton_keys / structural_voxel_key / point_feature_voxel_key の形状差を吸収する。
        """
        if unit_keys is None:
            return None

        if not torch.is_tensor(unit_keys):
            unit_keys = torch.as_tensor(unit_keys)

        unit_keys = unit_keys.to(device=device, dtype=torch.long)

        if unit_keys.ndim == 1:
            unit_keys = unit_keys.view(1, -1)

        elif unit_keys.ndim == 2:
            if unit_keys.shape[0] == 1:
                pass
            elif unit_keys.shape[1] == 1:
                unit_keys = unit_keys.reshape(1, -1)
            else:
                pass

        elif unit_keys.ndim == 3:
            if unit_keys.shape[1] == 1:
                unit_keys = unit_keys.squeeze(1)
            elif unit_keys.shape[2] == 1:
                unit_keys = unit_keys.squeeze(2)
            else:
                return None
        else:
            return None

        if unit_keys.ndim != 2:
            return None

        B = int(batch_size)
        N = int(point_count)

        if unit_keys.shape[0] == 1 and B > 1:
            unit_keys = unit_keys.expand(B, -1).contiguous()

        if unit_keys.shape[0] != B:
            return None

        current_n = int(unit_keys.shape[1])

        if current_n == N:
            return unit_keys.contiguous()

        if current_n <= 0:
            return None

        if current_n > N:
            return unit_keys[:, :N].contiguous()

        pad = unit_keys[:, -1:].expand(B, N - current_n)
        return torch.cat([unit_keys, pad], dim=1).contiguous()

    def _unit_keys_from_voxel_coords(self, context, batch_size, point_count, device):
        """
        repair_unit_keys / global_morton_keys が無い場合に、
        global_voxel_coords から CauseDiagnosisAggregation 用の unit_keys=[B,N] を作る。
        これは local recompute ではなく、prebuilt global voxel coords に基づく fallback である。
        """
        if not isinstance(context, dict):
            return None

        coords_raw = self._first_tensor_from_dict(
            context,
            (
                "global_voxel_coords",
                "subtree_global_voxel_coords",
                "occupied_voxel_coords",
                "full_global_voxel_coords",
                "full_occupied_voxel_coords",
            ),
        )
        if coords_raw is None:
            return None

        coords_b3n = self._normalize_node_voxel_coords(coords_raw, device=device)
        if coords_b3n is None or coords_b3n.shape[-1] <= 0:
            return None

        coords_b3n = coords_b3n.to(device=device, dtype=torch.long)

        if coords_b3n.shape[0] == 1 and int(batch_size) > 1:
            coords_b3n = coords_b3n.expand(int(batch_size), -1, -1).contiguous()

        if coords_b3n.shape[0] != int(batch_size):
            return None

        # 点数を analysis_xyz に合わせる。
        current_n = int(coords_b3n.shape[-1])
        target_n = int(point_count)

        if current_n > target_n:
            coords_b3n = coords_b3n[:, :, :target_n].contiguous()
        elif current_n < target_n:
            if current_n <= 0:
                return None
            pad = coords_b3n[:, :, -1:].expand(coords_b3n.shape[0], 3, target_n - current_n)
            coords_b3n = torch.cat([coords_b3n, pad], dim=2).contiguous()

        # 3次元voxel座標から安定した整数keyを作る。
        # morton keyそのものではないが、同一voxelを同一repair unitにまとめる目的には十分である。
        unit_keys = (
            coords_b3n[:, 0, :] * 73856093
            + coords_b3n[:, 1, :] * 19349663
            + coords_b3n[:, 2, :] * 83492791
        )

        return self._normalize_unit_keys(
            unit_keys,
            batch_size=batch_size,
            point_count=point_count,
            device=device,
        )
    
    @staticmethod
    def _first_tensor_from_dict(dict_obj, keys):
        if not isinstance(dict_obj, dict):
            return None
        for key in keys:
            value = dict_obj.get(key, None)
            if torch.is_tensor(value):
                return value
        return None

    @staticmethod
    def _first_value_from_dict(dict_obj, keys):
        if not isinstance(dict_obj, dict):
            return None
        for key in keys:
            if key in dict_obj and dict_obj.get(key, None) is not None:
                return dict_obj.get(key)
        return None

    @staticmethod
    def _child_slot_from_coords_lastdim(coords):
        coords = coords.to(dtype=torch.long)
        return (
            coords[..., 0].remainder(2)
            + 2 * coords[..., 1].remainder(2)
            + 4 * coords[..., 2].remainder(2)
        ).to(dtype=torch.long)

    def _parent_occupancy_codes_and_child_slots(self, coords_b3n):
        B, _, N = coords_b3n.shape
        parent_codes = torch.zeros((B, N), device=coords_b3n.device, dtype=torch.long)
        child_slots = torch.zeros((B, N), device=coords_b3n.device, dtype=torch.long)
        pattern_weights = (2 ** torch.arange(8, device=coords_b3n.device, dtype=torch.long)).view(1, 8)
        for b in range(B):
            coords = coords_b3n[b].transpose(0, 1).contiguous().to(dtype=torch.long)
            if coords.numel() == 0:
                continue
            parents = torch.div(coords, 2, rounding_mode="floor")
            slots = self._child_slot_from_coords_lastdim(coords)
            child_slots[b] = slots
            unique_parents, inverse = torch.unique(parents, dim=0, sorted=True, return_inverse=True)
            occupancy = torch.zeros(
                (unique_parents.shape[0], 8),
                device=coords_b3n.device,
                dtype=torch.bool,
            )
            occupancy[inverse, slots.clamp(0, 7)] = True
            codes = (occupancy.to(dtype=torch.long) * pattern_weights).sum(dim=1)
            parent_codes[b] = codes.index_select(0, inverse)
        return parent_codes, child_slots

    def _occupancy_code_popularity_for_features(self, like_tensor, *contexts):
        parts = []

        def _add_codes(ctx, key, weight=1.0):
            if not isinstance(ctx, dict) or key not in ctx:
                return
            value = ctx.get(key, None)
            if value is None:
                return
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(device=like_tensor.device, dtype=torch.long).reshape(-1)
            value = value[(value >= 0) & (value < 256)]
            if value.numel() <= 0:
                return
            repeat = max(int(round(float(weight))), 1)
            parts.append(value.repeat(repeat))

        for ctx in contexts:
            _add_codes(ctx, "occupancy_codes", weight=3.0)
            _add_codes(ctx, "ancestor_occupancy_codes", weight=2.0)
            _add_codes(ctx, "sibling_occupancy_codes", weight=2.0)
            _add_codes(ctx, "parent_occupancy_code", weight=2.0)

        smoothing = max(float(getattr(self.args, "repair_pattern_prior_smoothing", 1.0)), 0.0)
        counts = like_tensor.new_ones((256,), dtype=torch.float32) * smoothing
        if parts:
            codes = torch.cat(parts, dim=0)
            counts.scatter_add_(0, codes, torch.ones_like(codes, dtype=counts.dtype))
        popularity = torch.log1p(counts)
        return popularity / popularity.amax().clamp_min(1e-6)

    def _fit_feature_channels(self, feature, target_channels):
        if feature.shape[1] == int(target_channels):
            return feature
        if feature.shape[1] > int(target_channels):
            return feature[:, :int(target_channels), :].contiguous()
        pad = feature.new_zeros(
            feature.shape[0],
            int(target_channels) - feature.shape[1],
            feature.shape[2],
        )
        return torch.cat([feature, pad], dim=1).contiguous()

    def _build_node_features_from_voxel_coords(
        self,
        coords_b3n,
        pts_xyz,
        node_mask=None,
        subtree_tree=None,
        full_octree_context=None,
    ):
        coords_f = coords_b3n.to(device=pts_xyz.device, dtype=pts_xyz.dtype)
        if coords_f.shape[-1] <= 0:
            return pts_xyz.new_zeros((pts_xyz.shape[0], int(getattr(self.args, "fused_feat_dim", getattr(self.args, "out_dim", 64))), 0))

        if node_mask is not None and torch.is_tensor(node_mask):
            mask = node_mask.to(device=pts_xyz.device, dtype=torch.bool)
            if mask.ndim == 3:
                mask = mask.squeeze(1)
        else:
            mask = torch.ones((coords_f.shape[0], coords_f.shape[-1]), device=pts_xyz.device, dtype=torch.bool)

        valid_f = mask.unsqueeze(1).to(dtype=coords_f.dtype)
        denom = valid_f.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (coords_f * valid_f).sum(dim=2, keepdim=True) / denom
        centered = coords_f - mean
        span = centered.abs().amax(dim=2, keepdim=True).clamp_min(1.0)
        norm = centered / span

        radius = torch.linalg.norm(norm, dim=1, keepdim=True).clamp(0.0, 4.0) / 4.0
        parity = (coords_b3n.remainder(2).to(dtype=pts_xyz.dtype) * 2.0) - 1.0
        depth_proxy = torch.linalg.norm(coords_f - coords_f.amin(dim=2, keepdim=True), dim=1, keepdim=True)
        depth_proxy = depth_proxy / depth_proxy.amax(dim=2, keepdim=True).clamp_min(1.0)

        parent_codes, child_slots = self._parent_occupancy_codes_and_child_slots(coords_b3n)
        slot_values = torch.arange(8, device=coords_b3n.device, dtype=torch.long).view(1, 8, 1)
        child_slot_onehot = (child_slots.unsqueeze(1) == slot_values).to(dtype=pts_xyz.dtype)
        parent_code_feature = (parent_codes.to(dtype=pts_xyz.dtype).unsqueeze(1) / 255.0) * 2.0 - 1.0
        child_count = torch.zeros_like(parent_codes, dtype=pts_xyz.dtype)
        for bit in range(8):
            child_count = child_count + ((parent_codes >> bit) & 1).to(dtype=pts_xyz.dtype)
        child_count = (child_count.unsqueeze(1) / 8.0).clamp(0.0, 1.0)
        code_popularity = self._occupancy_code_popularity_for_features(
            pts_xyz,
            subtree_tree,
            full_octree_context,
        )
        parent_popularity = code_popularity.index_select(
            0,
            parent_codes.reshape(-1).clamp(0, 255),
        ).view(coords_b3n.shape[0], coords_b3n.shape[-1]).unsqueeze(1).to(dtype=pts_xyz.dtype)

        base_feature = torch.cat(
            [
                norm,
                norm.abs(),
                radius,
                parity,
                depth_proxy,
                child_slot_onehot,
                parent_code_feature,
                child_count,
                parent_popularity,
                valid_f,
            ],
            dim=1,
        )
        return self._fit_feature_channels(
            base_feature,
            int(getattr(self.args, "fused_feat_dim", getattr(self.args, "out_dim", 64))),
        )

    def _build_node_voxel_input(
        self,
        pts_xyz,
        coord_scale=None,
        subtree_tree=None,
        full_octree_context=None,
        octree_input_mode="auto",
    ):
        """
        subtree_tree/full_octree_context/canonical voxel coordsから
        Network用のnode/voxel featureを作る。
        """
        debug = {
            "network_voxel_node_input_requested": bool(getattr(self.args, "network_voxel_node_input", False)),
            "network_voxel_node_input_used": False,
            "network_voxel_node_fallback": False,
            "network_voxel_node_fallback_reason": "",
            "network_voxel_node_count": 0,
            "network_voxel_node_source": "none",
            "network_voxel_node_feature_shape": "",
        }

        if not bool(getattr(self.args, "network_voxel_node_input", False)):
            debug["network_voxel_node_fallback_reason"] = "disabled_by_args"
            return None, debug

        allow_fallback = bool(getattr(self.args, "network_voxel_node_fallback_point", True))
        use_subtree = bool(getattr(self.args, "voxel_node_use_subtree_context", True))
        use_full = bool(getattr(self.args, "voxel_node_use_full_context", True))

        coords_raw = None
        source = "none"

        if use_subtree and isinstance(subtree_tree, dict):
            coords_raw = self._first_tensor_from_dict(
                subtree_tree,
                (
                    "global_voxel_coords",
                    "subtree_global_voxel_coords",
                    "occupied_voxel_coords",
                    "full_global_voxel_coords",
                ),
            )
            if coords_raw is not None:
                source = "subtree_tree"

        if coords_raw is None and use_full and isinstance(full_octree_context, dict):
            coords_raw = self._first_tensor_from_dict(
                full_octree_context,
                (
                    "global_voxel_coords",
                    "full_global_voxel_coords",
                    "full_occupied_voxel_coords",
                    "occupied_voxel_coords",
                ),
            )
            if coords_raw is not None:
                source = "full_octree_context"

        meta = None
        if coords_raw is None:
            missing_subtree_and_full_context = (
                subtree_tree is None
                and full_octree_context is None
            )

            # full_octree_context がある場合は、subtree_tree が無くてもNode/Voxel経路を継続する。
            # full cloud anchorでは full_octree_context の global_voxel_coords を使う。
            if missing_subtree_and_full_context and allow_fallback:
                debug["network_voxel_node_fallback"] = True
                debug["network_voxel_node_fallback_reason"] = "missing_subtree_and_full_context"
                return None, debug

            try:
                q_result = quantize_sparsepcgc_coords(
                    pts_xyz,
                    self.args,
                    coord_scale=coord_scale,
                    offset=None,
                    return_metadata=True,
                )
                if isinstance(q_result, tuple) and len(q_result) == 2:
                    coords_raw, meta = q_result
                else:
                    coords_raw = q_result
                source = "canonical_quantize"
            except Exception as exc:
                if allow_fallback:
                    debug["network_voxel_node_fallback"] = True
                    debug["network_voxel_node_fallback_reason"] = f"canonical_quantize_failed:{type(exc).__name__}"
                    return None, debug
                raise

        coords_b3n = self._normalize_node_voxel_coords(coords_raw, device=pts_xyz.device)
        if coords_b3n is None or coords_b3n.shape[-1] <= 0:
            if allow_fallback:
                debug["network_voxel_node_fallback"] = True
                debug["network_voxel_node_fallback_reason"] = "invalid_or_empty_voxel_coords"
                return None, debug
            raise ValueError("Node/Voxel input requested but voxel coords are invalid or empty.")

        unique_result = unique_voxel_coords_batched(coords_b3n)
        voxel_coords = unique_result["coords"].to(device=pts_xyz.device, dtype=torch.long)
        node_mask = unique_result["valid_mask"].to(device=pts_xyz.device, dtype=torch.bool)
        node_counts = unique_result["counts"].to(device=pts_xyz.device, dtype=torch.long)

        if isinstance(meta, dict):
            restore_meta = dict(meta)
        else:
            restore_meta = {}

        context_for_meta = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        if isinstance(context_for_meta, dict):
            global_qs = self._first_value_from_dict(context_for_meta, ("global_qs", "effective_qs"))
            global_offset = self._first_value_from_dict(context_for_meta, ("global_offset", "global_offset_tensor"))
            if global_qs is not None:
                restore_meta["global_qs"] = global_qs
            if global_offset is not None:
                restore_meta["global_offset"] = global_offset

        try:
            node_xyz, restore_info = restore_points_from_voxel_coords(
                voxel_coords,
                meta=restore_meta if restore_meta else None,
                args=self.args,
                center=bool(getattr(self.args, "sparsepcgc_dequantize_center", False)),
                unique=False,
                dtype=pts_xyz.dtype,
                device=pts_xyz.device,
            )
        except Exception as exc:
            if allow_fallback:
                debug["network_voxel_node_fallback"] = True
                debug["network_voxel_node_fallback_reason"] = f"restore_failed:{type(exc).__name__}"
                return None, debug
            raise

        node_features = self._build_node_features_from_voxel_coords(
            voxel_coords,
            pts_xyz=pts_xyz,
            node_mask=node_mask,
            subtree_tree=subtree_tree,
            full_octree_context=full_octree_context,
        )

        debug.update(
            {
                "network_voxel_node_input_used": True,
                "network_voxel_node_fallback": False,
                "network_voxel_node_fallback_reason": "",
                "network_voxel_node_count": int(node_xyz.shape[-1]),
                "network_voxel_node_source": str(source),
                "network_voxel_node_feature_shape": str(tuple(node_features.shape)),
            }
        )

        return {
            "voxel_coords": voxel_coords,
            "node_xyz": node_xyz,
            "node_features": node_features,
            "node_mask": node_mask,
            "node_counts": node_counts,
            "global_qs": restore_meta.get("global_qs", None),
            "global_offset": restore_meta.get("global_offset", None),
            "restore_meta": restore_meta,
            "restore_info": restore_info,
            "source": str(source),
        }, debug

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
    def _slice_point_aligned_tensor(value, point_mask, batch_size, device):
        """
        full cloud 上の点・voxelに対応するTensorを、subtree maskで切り出す。
        ここではB=1のsubtree学習を主対象にする。
        """
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        value = value.to(device=device)
        mask = point_mask
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        mask = mask.to(device=device, dtype=torch.bool)

        if mask.shape[0] != batch_size:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            else:
                raise ValueError("subtree point_mask batch size does not match.")

        # B=1以外では可変長subsetをそのままTensor化できないため、明示的に止める。
        if batch_size != 1:
            raise ValueError(
                "full-cloud-coordinate subtree subset currently supports batch_size=1 only. "
                "Use batch_size=1 for subtree training or add padding logic."
            )

        mask_b = mask[0]

        # [B, 3, N] または [B, C, N]
        if value.ndim == 3 and value.shape[0] in (1, batch_size) and value.shape[2] == mask_b.numel():
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1, -1)
            return value[0:1, :, mask_b].contiguous()

        # [B, N, K]
        if value.ndim == 3 and value.shape[0] in (1, batch_size) and value.shape[1] == mask_b.numel():
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1, -1)
            return value[0:1, mask_b, :].contiguous()

        # [B, N]
        if value.ndim == 2 and value.shape[0] in (1, batch_size) and value.shape[1] == mask_b.numel():
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            return value[0:1, mask_b].contiguous()

        # [N]
        if value.ndim == 1 and value.shape[0] == mask_b.numel():
            return value[mask_b].view(1, -1).contiguous()

        # 点数に対応しないメタ情報はそのまま返す。
        return value

    def _build_full_coord_subtree_context(
        self,
        base_context,
        point_mask,
        batch_size,
        device,
    ):
        """
        full cloud contextから、selected_subtree_keysに対応する点だけを切り出した
        subtree contextを作る。

        重要：
        global_voxel_coordsを再計算しない。
        full cloudで作ったglobal_voxel_coordsのsubsetだけを使う。
        """
        if not isinstance(base_context, dict):
            return None

        subtree_context = {}

        point_aligned_keys = (
            "global_voxel_coords",
            "subtree_global_voxel_coords",
            "occupied_voxel_coords",
            "full_global_voxel_coords",
            "repair_unit_keys",
            "point_node_ids",
            "point_subtree_keys",
            "point_parent_node_ids",
            "point_child_slots",
            "point_valid_empty_child_mask",
        )

        for key, value in base_context.items():
            if key in point_aligned_keys:
                subtree_context[key] = self._slice_point_aligned_tensor(
                    value,
                    point_mask=point_mask,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                subtree_context[key] = value

        # 以後の全モジュールが同じ名前を見るように正規化する。
        if subtree_context.get("global_voxel_coords", None) is None:
            for alt_key in ("subtree_global_voxel_coords", "occupied_voxel_coords", "full_global_voxel_coords"):
                if subtree_context.get(alt_key, None) is not None:
                    subtree_context["global_voxel_coords"] = subtree_context[alt_key]
                    break

        subtree_context["subtree_is_full_cloud_coord_subset"] = True
        return subtree_context

    @staticmethod
    def _fit_point_key_rows(keys, batch_size, num_points, device):
        if keys is None:
            return None
        keys = keys.to(device=device, dtype=torch.long)
        if keys.ndim == 1:
            keys = keys.view(1, -1)
        elif keys.ndim == 3 and keys.shape[1] == 1:
            keys = keys.squeeze(1)
        if keys.ndim != 2:
            return None
        if keys.shape[0] == 1 and batch_size > 1:
            keys = keys.expand(batch_size, -1)
        if keys.shape[0] != batch_size:
            return None
        current = int(keys.shape[1])
        if current == num_points:
            return keys
        if current <= 0:
            return torch.zeros((batch_size, num_points), device=device, dtype=torch.long)
        if current > num_points:
            return keys[:, :num_points]
        pad = keys[:, -1:].expand(batch_size, num_points - current)
        return torch.cat([keys, pad], dim=1)

    def _tree_point_keys(self, subtree_tree, key_names, batch_size, num_points, device):
        if not isinstance(subtree_tree, dict):
            return None
        for key_name in key_names:
            value = subtree_tree.get(key_name, None)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            fitted = self._fit_point_key_rows(value, batch_size, num_points, device)
            if fitted is not None:
                return fitted
        return None

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

    def _maybe_fast_full_cloud_oracle_forward(
        self,
        pts_xyz,
        compute_internal_losses,
        full_octree_context,
        octree_input_mode,
    ):
        """
        no-gradのFullCloud anchorで、actual oracle full overrideを採択済みなら、
        巨大なNetwork/Actuator forwardを実測済みvoxel stateのセットだけに短絡する。
        actual候補encodeとshadow subtreeの学習経路はそのまま残す。
        """
        if not bool(getattr(self.args, "fast_full_cloud_oracle_anchor", True)):
            return None
        if compute_internal_losses is not False:
            return None
        if str(octree_input_mode or "").strip().lower() != "full_cloud":
            return None
        if not isinstance(full_octree_context, dict):
            return None
        if not bool(full_octree_context.get("fast_full_cloud_oracle_anchor", False)):
            return None
        if str(full_octree_context.get("actual_oracle_override_scope", "")) != "full_cloud":
            return None

        override_coords = full_octree_context.get("actual_oracle_override_final_voxel_coords", None)
        if not torch.is_tensor(override_coords):
            return None
        coords_b3n = self._normalize_node_voxel_coords(override_coords, device=pts_xyz.device)
        if coords_b3n is None or coords_b3n.ndim != 3 or coords_b3n.shape[1] != 3 or coords_b3n.shape[-1] <= 0:
            return None

        before_coords = full_octree_context.get("full_global_voxel_coords", None)
        if before_coords is None:
            before_coords = full_octree_context.get("global_voxel_coords", None)
        before_b3n = self._normalize_node_voxel_coords(before_coords, device=pts_xyz.device)
        before_count = int(before_b3n.shape[-1]) if torch.is_tensor(before_b3n) else int(pts_xyz.shape[-1])
        after_count = int(coords_b3n.shape[-1])

        valid_mask = torch.ones(
            (coords_b3n.shape[0], coords_b3n.shape[-1]),
            device=pts_xyz.device,
            dtype=torch.bool,
        )
        voxel_state = {
            "initial_voxel_coords": before_b3n.detach() if torch.is_tensor(before_b3n) else coords_b3n.detach(),
            "final_voxel_coords": coords_b3n.detach(),
            "final_voxel_valid_mask": valid_mask.detach(),
            "voxel_edit_mode": "actual_oracle_full_cloud_fast_anchor",
            "voxel_edit_state_enabled": True,
            "voxel_edit_initial_count": int(before_count),
            "voxel_edit_final_count": int(after_count),
            "input_voxel_count": int(before_count),
            "before_occupied_voxel_count": int(before_count),
            "final_voxel_count": int(after_count),
            "after_occupied_voxel_count": int(after_count),
            "voxel_edit_drop_count": int(full_octree_context.get("actual_oracle_override_drop_count", 0) or 0),
            "voxel_edit_add_count": int(full_octree_context.get("actual_oracle_accepted_add_count", 0) or 0),
            "voxel_edit_move_count": int(full_octree_context.get("actual_oracle_override_move_count", 0) or 0),
            "estimated_edit_record_bits": float(full_octree_context.get("actual_oracle_edit_record_bits", 0.0) or 0.0),
            "final_voxel_update_mode": "actual_oracle_full_cloud_override",
            "final_voxel_recomputed_from_pts_out": False,
            "actuator_voxel_mode": "actual_oracle_full_cloud_override",
            "actuator_local_recomputed": False,
            "actuator_octree_context_source": "full_octree_context",
            "actuator_octree_context_is_full_cloud": True,
            "actuator_octree_context_available": True,
            "actuator_octree_context_has_global_voxel_coords": True,
        }
        if "global_qs" in full_octree_context:
            voxel_state["voxel_step"] = full_octree_context["global_qs"]
        if "global_offset" in full_octree_context:
            voxel_state["voxel_offset"] = full_octree_context["global_offset"]

        self.last_actuator_voxel_state = voxel_state
        try:
            setattr(self.args, "_last_actuator_voxel_state", self.last_actuator_voxel_state)
        except Exception:
            pass

        zero = pts_xyz.new_zeros(())
        final_w = pts_xyz.new_ones((pts_xyz.shape[0], 1, pts_xyz.shape[-1]))
        out_label = {
            "full_cloud_oracle_fast_path": True,
            "canonical_voxel_coords_before": before_b3n.detach() if torch.is_tensor(before_b3n) else None,
            "canonical_voxel_coords_after": coords_b3n.detach(),
        }
        self.last_structure_debug = {
            "network_voxel_node_input_requested": bool(getattr(self.args, "network_voxel_node_input", False)),
            "network_voxel_node_input_used": False,
            "network_voxel_node_fallback": False,
            "network_voxel_node_fallback_reason": "fast_full_cloud_oracle_anchor",
            "network_voxel_node_count": int(after_count),
            "network_voxel_node_source": "actual_oracle_full_cloud_override",
            "full_cloud_oracle_fast_path": True,
        }
        self.last_runtime_timing = {
            "encode": 0.0,
            "structure": 0.0,
            "actuator": 0.0,
            "total_forward": 0.0,
            "fast_full_cloud_oracle_anchor": 1.0,
        }
        return pts_xyz, zero, zero, zero, final_w, zero, zero, zero, out_label

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

        self.last_actuator_voxel_state = None
        original_pts_xyz = pts_xyz
        original_pts_attr = pts_attr
        node_voxel_input_state = None
        node_voxel_debug = {
            "network_voxel_node_input_requested": bool(getattr(self.args, "network_voxel_node_input", False)),
            "network_voxel_node_input_used": False,
            "network_voxel_node_fallback": False,
            "network_voxel_node_fallback_reason": "not_evaluated",
            "network_voxel_node_count": 0,
            "network_voxel_node_source": "none",
            "network_voxel_node_feature_shape": "",
        }

        try:
            setattr(self.args, "_last_actuator_voxel_state", None)
        except Exception:
            pass

        fast_oracle_result = self._maybe_fast_full_cloud_oracle_forward(
            pts_xyz,
            compute_internal_losses,
            full_octree_context,
            octree_input_mode,
        )
        if fast_oracle_result is not None:
            return fast_oracle_result

        prebuilt_subtree_mode = subtree_tree is not None
        full_unit_keys = None # Subtree Key保存用の変数初期化
        selection_mask = None # 選択されたSubtreeに属する点だけを示すマスクの初期化
        canonical_subtree_tree = subtree_tree
        # full cloud forward でも、full_octree_context から repair unit key を先に用意する。
        # repair_unit_keys / point_node_ids / global_morton_keys が無い場合は、
        # global_voxel_coords から同一voxel単位の key を作る。
        if isinstance(full_octree_context, dict):
            full_unit_keys = self._tree_point_keys(
                full_octree_context,
                ("repair_unit_keys", "point_node_ids", "global_morton_keys"),
                pts_xyz.shape[0],
                pts_xyz.shape[2],
                pts_xyz.device,
            )
            if full_unit_keys is None:
                full_unit_keys = self._unit_keys_from_voxel_coords(
                    full_octree_context,
                    batch_size=pts_xyz.shape[0],
                    point_count=pts_xyz.shape[2],
                    device=pts_xyz.device,
                )
        prebuilt_repair_unit_keys = self._tree_point_keys(
            subtree_tree,
            ("repair_unit_keys", "point_node_ids"),
            pts_xyz.shape[0],
            pts_xyz.shape[2],
            pts_xyz.device,
        )
        prebuilt_subtree_keys = self._tree_point_keys(
            subtree_tree,
            ("point_subtree_keys",),
            pts_xyz.shape[0],
            pts_xyz.shape[2],
            pts_xyz.device,
        )
        if prebuilt_subtree_mode:
            full_unit_keys = prebuilt_repair_unit_keys

            if full_unit_keys is None:
                full_unit_keys = self._tree_point_keys(
                    subtree_tree,
                    ("global_morton_keys",),
                    pts_xyz.shape[0],
                    pts_xyz.shape[2],
                    pts_xyz.device,
                )

            if full_unit_keys is None:
                full_unit_keys = self._unit_keys_from_voxel_coords(
                    subtree_tree,
                    batch_size=pts_xyz.shape[0],
                    point_count=pts_xyz.shape[2],
                    device=pts_xyz.device,
                )

            if full_unit_keys is None:
                raise ValueError(
                    "subtree_tree must provide repair_unit_keys, point_node_ids, global_morton_keys, "
                    "or global_voxel_coords for Network prebuilt subtree mode."
                )
            if selected_subtree_keys is not None and prebuilt_subtree_keys is not None:
                selected_subtree_keys = selected_subtree_keys.to(device=pts_xyz.device, dtype=prebuilt_subtree_keys.dtype).reshape(-1)
                selection_mask = self._normalize_point_mask(
                    subtree_membership_mask(prebuilt_subtree_keys, selected_subtree_keys),
                    batch_size=pts_xyz.shape[0],
                    num_points=pts_xyz.shape[2],
                    device=pts_xyz.device,
                )
                canonical_subtree_tree = self._build_full_coord_subtree_context(
                    subtree_tree,
                    point_mask=selection_mask,
                    batch_size=pts_xyz.shape[0],
                    device=pts_xyz.device,
                )
                # canonical_subtree_tree に含まれる点数に合わせて key を取り直す。
                canonical_point_count = pts_xyz.shape[2]
                if isinstance(canonical_subtree_tree, dict):
                    gv = canonical_subtree_tree.get("global_voxel_coords", None)
                    gv_norm = self._normalize_node_voxel_coords(gv, device=pts_xyz.device)
                    if gv_norm is not None:
                        canonical_point_count = int(gv_norm.shape[-1])

                full_unit_keys = self._tree_point_keys(
                    canonical_subtree_tree,
                    ("repair_unit_keys", "point_node_ids", "global_morton_keys"),
                    pts_xyz.shape[0],
                    canonical_point_count,
                    pts_xyz.device,
                )
                if full_unit_keys is None:
                    full_unit_keys = self._unit_keys_from_voxel_coords(
                        canonical_subtree_tree,
                        batch_size=pts_xyz.shape[0],
                        point_count=canonical_point_count,
                        device=pts_xyz.device,
                    )
        elif subtree_ref is not None:
            # full_octree_contextにpoint_subtree_keysがあるなら、それを優先する。
            # なければ従来通りpts_xyzから計算するが、これはfallbackである。
            full_context_subtree_keys = self._tree_point_keys(
                full_octree_context,
                ("point_subtree_keys",),
                pts_xyz.shape[0],
                pts_xyz.shape[2],
                pts_xyz.device,
            )

            full_context_repair_keys = self._tree_point_keys(
                full_octree_context,
                ("repair_unit_keys", "point_node_ids", "global_morton_keys"),
                pts_xyz.shape[0],
                pts_xyz.shape[2],
                pts_xyz.device,
            )

            if full_context_repair_keys is not None:
                full_unit_keys = full_context_repair_keys
            elif full_context_subtree_keys is not None:
                # 最後のfallbackとしてのみ使う。
                # これはsubtree単位の粗い集約になるため、repair unitとしては精度が落ちる。
                full_unit_keys = full_context_subtree_keys
            else:
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

                if isinstance(full_octree_context, dict):
                    canonical_subtree_tree = self._build_full_coord_subtree_context(
                        full_octree_context,
                        point_mask=selection_mask,
                        batch_size=pts_xyz.shape[0],
                        device=pts_xyz.device,
                    )

        timing_enabled = self._timing_enabled() # 時間計測を行うか否か取得
        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_t0 = time.time()
            
        """Encoder"""
        if bool(getattr(self.args, "network_voxel_node_input", False)):
            node_voxel_input_state, node_voxel_debug = self._build_node_voxel_input(
                pts_xyz,
                coord_scale=coord_scale,
                subtree_tree=canonical_subtree_tree,
                full_octree_context=full_octree_context,
                octree_input_mode=octree_input_mode,
            )
            if node_voxel_input_state is not None:
                pts_xyz = node_voxel_input_state["node_xyz"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                pts_attr = None

                # ここで selection_mask を full cloud 用maskのまま使ってはいけない。
                # pts_xyz はすでに subtree subset のnode_xyzになっているため、
                # 以後のselection_maskは subset 内の有効node全体を指すmaskにする。
                selection_mask = node_voxel_input_state["node_mask"]

                # subset化済みcontextを以後の構造解析・Actuatorにも渡す。
                subtree_tree = canonical_subtree_tree

                prebuilt_subtree_mode = False
                full_unit_keys = None
                prebuilt_repair_unit_keys = None
                prebuilt_subtree_keys = None

        # ============================================================
        # Phase4:
        # CostAttributionModule に、現在の入力が点群かNode/Voxelかを伝える。
        # これを入れないと cost_attribution.py 側の input_mode debug が常に point になりやすい。
        # ============================================================
        try:
            self.cost_attributor.node_voxel_mode = bool(node_voxel_input_state is not None)
        except Exception:
            pass
        """Encoder"""
        if node_voxel_input_state is not None:
            fused_feat_node = node_voxel_input_state["node_features"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            encode_state = {
                "fused_feat": fused_feat_node,
                "local_feat": fused_feat_node,
                "analysis_xyz": pts_xyz,
                "analysis_counts": [
                    int(count.detach().cpu()) if torch.is_tensor(count) else int(count)
                    for count in node_voxel_input_state["node_counts"]
                ],
                "encoder_counts": [
                    int(count.detach().cpu()) if torch.is_tensor(count) else int(count)
                    for count in node_voxel_input_state["node_counts"]
                ],
                "full_counts": [
                    int(count.detach().cpu()) if torch.is_tensor(count) else int(count)
                    for count in node_voxel_input_state["node_counts"]
                ],
                "raw_counts": [
                    int(original_pts_xyz.shape[-1])
                    for _ in range(int(pts_xyz.shape[0]))
                ],
                "pre_sparse_counts": [
                    int(count.detach().cpu()) if torch.is_tensor(count) else int(count)
                    for count in node_voxel_input_state["node_counts"]
                ],
                "voxel_sizes": [0.0 for _ in range(int(pts_xyz.shape[0]))],
                "kept_sparse_after_encoder": False,
            }
        else:
            encode_state = self._encode(pts_xyz, coord_scale=coord_scale)

        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_encode_end = time.time()
        fused_feat = encode_state["fused_feat"] # 統合抽出
        analysis_xyz = encode_state["analysis_xyz"] # Octree構造解析に使う点群座標を取り出す
        analysis_counts = encode_state["analysis_counts"] # analysis_xyzの有効点数をバッチごとに取り出す
        keep_sparse_path = bool(encode_state.get("kept_sparse_after_encoder", False))
        if node_voxel_input_state is not None:
            analysis_unit_keys = None
            analysis_subtree_keys = None
            if isinstance(node_voxel_input_state.get("voxel_coords", None), torch.Tensor):
                coords_for_keys = node_voxel_input_state["voxel_coords"]
                if coords_for_keys.ndim == 3 and coords_for_keys.shape[1] == 3:
                    coords_key = coords_for_keys[:, 0, :].to(torch.long) * 73856093
                    coords_key = coords_key + coords_for_keys[:, 1, :].to(torch.long) * 19349663
                    coords_key = coords_key + coords_for_keys[:, 2, :].to(torch.long) * 83492791
                    analysis_unit_keys = coords_key
                    analysis_subtree_keys = coords_key

        elif prebuilt_subtree_mode:
            analysis_unit_keys = self._tree_point_keys(
                subtree_tree,
                ("repair_unit_keys", "point_node_ids"),
                analysis_xyz.shape[0],
                analysis_xyz.shape[2],
                analysis_xyz.device,
            )
            analysis_subtree_keys = self._tree_point_keys(
                subtree_tree,
                ("point_subtree_keys",),
                analysis_xyz.shape[0],
                analysis_xyz.shape[2],
                analysis_xyz.device,
            )

        else:
            analysis_unit_keys = assign_octree_subtree_keys(analysis_xyz, subtree_ref) if subtree_ref is not None else None
            analysis_subtree_keys = analysis_unit_keys

        analysis_selection_mask = (
            subtree_membership_mask(analysis_subtree_keys, selected_subtree_keys)
            if analysis_subtree_keys is not None and selected_subtree_keys is not None
            else None
        )
        if timing_enabled:
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
            occupancy_nll_proxy_full_list = []
            quant_proxy_full_list = []
            cause_scores_means = []
            subtree_scores_means = []
            policy_probs_means = []
            aggregation_unit_modes = []
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
                    subtree_tree=canonical_subtree_tree if b == 0 else None,
                    full_octree_context=full_octree_context if b == 0 else None,
                    octree_input_mode=octree_input_mode,
                )
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
                unit_keys_b = None if analysis_unit_keys is None else analysis_unit_keys[b:b + 1, :analysis_count]
                aggregation_key_source_b = "analysis_unit_keys" if unit_keys_b is not None else "none"

                if unit_keys_b is None:
                    unit_keys_b = structure_b.get("structural_voxel_key", None)
                    if unit_keys_b is not None:
                        aggregation_key_source_b = "structure_b.structural_voxel_key"

                    if unit_keys_b is None:
                        unit_keys_b = structure_b.get("point_feature_voxel_key", None)
                        if unit_keys_b is not None:
                            aggregation_key_source_b = "structure_b.point_feature_voxel_key"

                unit_keys_b = self._normalize_unit_keys(
                    unit_keys_b,
                    batch_size=analysis_xyz_b.shape[0],
                    point_count=analysis_xyz_b.shape[-1],
                    device=analysis_xyz_b.device,
                )

                if unit_keys_b is None:
                    raise ValueError(
                        "Network.forward could not provide prebuilt unit_keys to CauseDiagnosisAggregation "
                        f"in sparse/per-sample path. analysis_xyz_b={tuple(analysis_xyz_b.shape)}, "
                        f"structural_voxel_key={None if structure_b.get('structural_voxel_key', None) is None else tuple(structure_b.get('structural_voxel_key').shape)}, "
                        f"point_feature_voxel_key={None if structure_b.get('point_feature_voxel_key', None) is None else tuple(structure_b.get('point_feature_voxel_key').shape)}"
                    )

                aggregated_b = self.cause_aggregator(
                    pts_xyz=analysis_xyz_b,
                    cause_scores=cause_scores_b,
                    cause_targets=cause_targets_b,
                    unit_keys=unit_keys_b,
                )
                aggregation_unit_modes.append(str(aggregated_b.get("unit_mode", "unknown")))
                if isinstance(structure_b, dict):
                    structure_b["phase4_aggregation_key_source"] = str(aggregation_key_source_b)
                    structure_b["phase4_aggregation_unit_count"] = int(
                        aggregated_b.get("unit_count", 0)
                    )
                    structure_b["phase4_aggregation_max_unit_size"] = int(
                        aggregated_b.get("max_unit_size", 0)
                    )
                    structure_b["phase4_aggregation_min_unit_size"] = int(
                        aggregated_b.get("min_unit_size", 0)
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
                    lowprob_proxy_full_list.append(structure_b["lowprob_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
                    occupancy_nll_proxy_full_list.append(structure_b["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype))
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
                    lowprob_proxy_b = structure_b["lowprob_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    occupancy_nll_proxy_b = structure_b["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    quant_proxy_b = structure_b["quant_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
                    single_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, single_proxy_b))
                    node_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, node_proxy_b))
                    lowprob_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, lowprob_proxy_b))
                    quant_proxy_full_list.append(self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, quant_proxy_b))
                    occupancy_nll_proxy_full_list.append(
                        self._propagate_encoder_features(full_xyz_b, analysis_xyz_b, occupancy_nll_proxy_b)
                    )

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
                "phase4_aggregation_key_source": structure_b.get(
                    "phase4_aggregation_key_source",
                    "unknown",
                ) if structure_b is not None else "unknown",
                "phase4_aggregation_unit_count": int(
                    structure_b.get("phase4_aggregation_unit_count", 0)
                    if structure_b is not None
                    else 0
                ),
                "phase4_aggregation_max_unit_size": int(
                    structure_b.get("phase4_aggregation_max_unit_size", 0)
                    if structure_b is not None
                    else 0
                ),
                "phase4_aggregation_min_unit_size": int(
                    structure_b.get("phase4_aggregation_min_unit_size", 0)
                    if structure_b is not None
                    else 0
                ),
                "phase4_structural_key_source": structure_b.get(
                    "phase4_structural_key_source",
                    "unknown",
                ) if structure_b is not None else "unknown",
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
                "occupancy_nll_proxy_full": torch.cat(occupancy_nll_proxy_full_list, dim=0),
                "leaf_pattern_diag": structure_b.get("leaf_pattern_diag") if structure_b is not None else None,

                # Section2:
                # sparse pathでは最後に処理したstructure_bのsummaryをdebugとして持つ。
                "leaf_pattern_available": bool(structure_b.get("leaf_pattern_available", False)) if structure_b is not None else False,
                "leaf_pattern_source": str(structure_b.get("leaf_pattern_source", "none")) if structure_b is not None else "none",
                "leaf_pattern_reason": str(structure_b.get("leaf_pattern_reason", "")) if structure_b is not None else "",
                "leaf_unique_parent_count": int(structure_b.get("leaf_unique_parent_count", 0) or 0) if structure_b is not None else 0,
                "leaf_unique_pattern_count": int(structure_b.get("leaf_unique_pattern_count", 0) or 0) if structure_b is not None else 0,
                "leaf_mean_child_count": float(structure_b.get("leaf_mean_child_count", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_single_child_parent_ratio": float(structure_b.get("leaf_single_child_parent_ratio", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_max_pattern_frequency": float(structure_b.get("leaf_max_pattern_frequency", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_candidate_available": bool(structure_b.get("leaf_candidate_available", False)) if structure_b is not None else False,
                "leaf_delete_gain_mean": float(structure_b.get("leaf_delete_gain_mean", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_add_gain_mean": float(structure_b.get("leaf_add_gain_mean", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_move_gain_mean": float(structure_b.get("leaf_move_gain_mean", 0.0) or 0.0) if structure_b is not None else 0.0,
                "leaf_high_gain_candidate_ratio": float(structure_b.get("leaf_high_gain_candidate_ratio", 0.0) or 0.0) if structure_b is not None else 0.0,
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
                subtree_tree=canonical_subtree_tree,
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
                
            # ============================================================
            # Phase4:
            # CauseAggregation に渡す unit_keys の出所を明示する。
            # local recomputeではなく、prebuilt/canonical keyを優先する。
            # ============================================================
            unit_keys = full_unit_keys
            aggregation_key_source = "full_unit_keys" if unit_keys is not None else "none"

            if unit_keys is None:
                unit_keys = structure.get("structural_voxel_key", None)
                if unit_keys is not None:
                    aggregation_key_source = "structure.structural_voxel_key"

            if unit_keys is None:
                unit_keys = structure.get("point_feature_voxel_key", None)
                if unit_keys is not None:
                    aggregation_key_source = "structure.point_feature_voxel_key"

            if unit_keys is None:
                unit_keys = self._unit_keys_from_voxel_coords(
                    canonical_subtree_tree,
                    batch_size=analysis_xyz.shape[0],
                    point_count=analysis_xyz.shape[-1],
                    device=analysis_xyz.device,
                )
                if unit_keys is not None:
                    aggregation_key_source = "canonical_subtree_tree.global_voxel_coords_hash"

            if unit_keys is None:
                unit_keys = self._unit_keys_from_voxel_coords(
                    full_octree_context,
                    batch_size=analysis_xyz.shape[0],
                    point_count=analysis_xyz.shape[-1],
                    device=analysis_xyz.device,
                )
                if unit_keys is not None:
                    aggregation_key_source = "full_octree_context.global_voxel_coords_hash"

            unit_keys = self._normalize_unit_keys(
                unit_keys,
                batch_size=analysis_xyz.shape[0],
                point_count=analysis_xyz.shape[-1],
                device=analysis_xyz.device,
            )

            if unit_keys is None:
                raise ValueError(
                    "Network.forward could not provide prebuilt unit_keys to CauseDiagnosisAggregation "
                    f"in full-cloud path. analysis_xyz={tuple(analysis_xyz.shape)}, "
                    f"full_unit_keys=None, "
                    f"structural_voxel_key={None if structure.get('structural_voxel_key', None) is None else tuple(structure.get('structural_voxel_key').shape)}, "
                    f"point_feature_voxel_key={None if structure.get('point_feature_voxel_key', None) is None else tuple(structure.get('point_feature_voxel_key').shape)}, "
                    f"octree_input_mode={structure.get('octree_input_mode', '')}, "
                    f"structural_voxel_mode={structure.get('structural_voxel_mode', '')}"
                )

            aggregated = self.cause_aggregator(
                pts_xyz=analysis_xyz,
                cause_scores=cause_scores,
                cause_targets=cause_targets,
                unit_keys=unit_keys,
            )

            structure["aggregation_unit_keys"] = unit_keys
            structure["aggregation_unit_mode"] = str(aggregated.get("unit_mode", "unknown"))

            # ============================================================
            # Phase4:
            # aggregation key の出所とunit統計をdebugへ保存する。
            # 同じ値を何度も代入しない。
            # ============================================================
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )

            aggregation_unit_modes = [str(aggregated.get("unit_mode", "unknown"))]
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            structure["phase4_aggregation_key_source"] = str(aggregation_key_source)
            structure["phase4_aggregation_unit_count"] = int(
                aggregated.get("unit_count", 0)
            )
            structure["phase4_aggregation_max_unit_size"] = int(
                aggregated.get("max_unit_size", 0)
            )
            structure["phase4_aggregation_min_unit_size"] = int(
                aggregated.get("min_unit_size", 0)
            )
            aggregation_unit_modes = [str(aggregated.get("unit_mode", "unknown"))]
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
            structure["lowprob_proxy_full"] = structure["lowprob_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            structure["occupancy_nll_proxy_full"] = structure["occupancy_nll_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            structure["quant_proxy_full"] = structure["quant_proxy"].to(device=pts_xyz.device, dtype=pts_xyz.dtype)
            loss_attr_sparse = None
            loss_policy_sparse = None

        if timing_enabled:
            self._sync_if_cuda_tensor(pts_xyz)
            runtime_structure_end = time.time()

        if node_voxel_input_state is not None and isinstance(structure, dict):
            structure["network_voxel_node_input_used"] = True
            structure["network_voxel_node_state"] = node_voxel_input_state
            structure["network_voxel_coords"] = node_voxel_input_state.get("voxel_coords", None)
            structure["network_voxel_node_mask"] = node_voxel_input_state.get("node_mask", None)
            structure["network_voxel_node_source"] = node_voxel_input_state.get("source", "unknown")
            if node_voxel_input_state.get("global_qs", None) is not None:
                structure["global_qs"] = node_voxel_input_state.get("global_qs", None)
            if node_voxel_input_state.get("global_offset", None) is not None:
                structure["global_offset"] = node_voxel_input_state.get("global_offset", None)

        """点操作実行"""
        actuator_input = torch.cat(
            [structure_feat_full, subtree_scores_full, policy_probs_full, repair_priority_full],
            dim=1,
        ) # 構造特徴、原因スコアなどをチャネル方向に結合

        # ============================================================
        # FullCloud時は、Actuatorにもfull_octree_contextをoctree_contextとして渡す。
        # これにより、Actuator内部のvoxel_coordsもfull cloud canonical基準に固定する。
        # ============================================================
        octree_mode_text = str(octree_input_mode or "auto").strip().lower()
        is_full_cloud_forward = octree_mode_text == "full_cloud"

        if is_full_cloud_forward:
            actuator_octree_context = full_octree_context
            actuator_octree_context_source = "full_octree_context"
        else:
            actuator_octree_context = canonical_subtree_tree
            actuator_octree_context_source = "canonical_subtree_tree"

        if (
            is_full_cloud_forward
            and bool(getattr(self.args, "full_cloud_require_actuator_octree_context", True))
            and not isinstance(actuator_octree_context, dict)
        ):
            raise RuntimeError(
                "FullCloud forward requires full_octree_context to be passed as "
                "Actuator octree_context, but full_octree_context is missing."
            )

        if (
            is_full_cloud_forward
            and bool(getattr(self.args, "full_cloud_require_actuator_octree_context", True))
            and isinstance(actuator_octree_context, dict)
            and actuator_octree_context.get("global_voxel_coords", None) is None
        ):
            raise RuntimeError(
                "FullCloud forward requires full_octree_context['global_voxel_coords'] "
                "for Actuator canonical voxel path."
            )

        pts_out, final_w, edit_loss, actuator_stats = self.actuator( # 実際に点操作を行う
            pts_xyz=pts_xyz,
            structure=structure,
            cause_scores=subtree_scores_full,
            policy_probs=policy_probs_full,
            actuator_features=actuator_input,
            repair_priority=repair_priority_full,
            coord_scale=coord_scale,
            selection_mask=selection_mask,
            octree_context=actuator_octree_context,
            full_octree_context=full_octree_context,
        )
        if isinstance(actuator_stats, dict):
            actuator_stats["actuator_octree_context_source"] = str(actuator_octree_context_source)
            actuator_stats["actuator_octree_context_is_full_cloud"] = bool(is_full_cloud_forward)
            actuator_stats["actuator_octree_context_available"] = bool(isinstance(actuator_octree_context, dict))
            actuator_stats["actuator_octree_context_has_global_voxel_coords"] = bool(
                isinstance(actuator_octree_context, dict)
                and actuator_octree_context.get("global_voxel_coords", None) is not None
            )
            
        # Actuatorが内部で更新したVoxel状態をLoss / compression.py 側から読めるように保存する。
        actuator_voxel_state = {
            key: actuator_stats.get(key, None)
            for key in (
                "initial_voxel_coords",
                "final_voxel_coords",
                "final_voxel_weights",
                "final_voxel_valid_mask",
                "voxel_step",
                "voxel_offset",
                "voxel_edit_mode",
                "voxel_edit_state_enabled",
                "voxel_edit_initial_count",
                "voxel_edit_final_count",
                "voxel_edit_drop_count",
                "voxel_edit_add_count",
                "voxel_edit_move_count",
                "voxel_edit_same_voxel_move_rejected",
                "voxel_edit_existing_target_rejected",
                "voxel_edit_duplicate_target_rejected",
                "voxel_edit_child_slot_rejected",
                "voxel_edit_empty_target_rejected",
                "point_aligned_initial_voxel_coords",
                "point_aligned_final_voxel_coords",
                "point_aligned_final_voxel_weights",
                "voxel_soft_drop_score",
                "voxel_soft_add_score",
                "voxel_soft_move_score",
                "voxel_soft_drop_amount",
                "voxel_soft_add_amount",
                "voxel_soft_move_amount",
                "voxel_soft_edit_score",
                "voxel_soft_edit_count_proxy",
                "drop_ratio_soft",
                "drop_ratio_hard",
                "add_ratio_soft",
                "add_ratio_hard",
                "move_ratio_soft",
                "move_ratio_hard",
                "add_ratio_loss_value",
                "add_consistency_loss_value",
                "voxel_soft_drop_mean",
                "voxel_soft_add_mean",
                "voxel_soft_move_mean",
                "operation_gate_prob",
                "operation_gate_hard",
                "operation_gate_logit",
                "drop_operation_gate",
                "add_operation_gate",
                "move_operation_gate",
                "raw_learned_drop_ratio",
                "raw_learned_add_ratio",
                "raw_learned_move_ratio",
            )
            if actuator_stats.get(key, None) is not None
        }
        if "voxel_edit_initial_count" in actuator_voxel_state:
            actuator_voxel_state["input_voxel_count"] = actuator_voxel_state["voxel_edit_initial_count"]
            actuator_voxel_state["before_occupied_voxel_count"] = actuator_voxel_state["voxel_edit_initial_count"]
        if "voxel_edit_final_count" in actuator_voxel_state:
            actuator_voxel_state["final_voxel_count"] = actuator_voxel_state["voxel_edit_final_count"]
            actuator_voxel_state["after_occupied_voxel_count"] = actuator_voxel_state["voxel_edit_final_count"]

        actuator_voxel_state["final_voxel_update_mode"] = actuator_stats.get(
            "final_voxel_update_mode",
            "unknown",
        )
        actuator_voxel_state["final_voxel_recomputed_from_pts_out"] = bool(
            actuator_stats.get("final_voxel_recomputed_from_pts_out", True)
        )
        actuator_voxel_state["actuator_voxel_mode"] = actuator_stats.get(
            "actuator_voxel_mode",
            "unknown",
        )
        actuator_voxel_state["actuator_local_recomputed"] = bool(
            actuator_stats.get("local_recomputed", actuator_stats.get("actuator_local_recomputed", True))
        )
        actuator_voxel_state["actuator_octree_context_source"] = str(
            actuator_stats.get("actuator_octree_context_source", "unknown")
        )
        actuator_voxel_state["actuator_octree_context_is_full_cloud"] = bool(
            actuator_stats.get("actuator_octree_context_is_full_cloud", False)
        )
        actuator_voxel_state["actuator_octree_context_available"] = bool(
            actuator_stats.get("actuator_octree_context_available", False)
        )
        actuator_voxel_state["actuator_octree_context_has_global_voxel_coords"] = bool(
            actuator_stats.get("actuator_octree_context_has_global_voxel_coords", False)
        )
        self.last_actuator_voxel_state = actuator_voxel_state
        try:
            setattr(self.args, "_last_actuator_voxel_state", self.last_actuator_voxel_state)
        except Exception:
            pass
        # Phase3: occupied voxel編集状態をdebug_tensorsからも参照できるようにする。
        for key in (
            "final_voxel_coords",
            "final_voxel_weights",
            "final_voxel_valid_mask",
            "voxel_edit_mode",
            "voxel_edit_state_enabled",
            "voxel_edit_initial_count",
            "voxel_edit_final_count",
            "voxel_edit_drop_count",
            "voxel_edit_add_count",
            "voxel_edit_move_count",
        ):
            if key in actuator_voxel_state:
                self.debug_tensors[key] = actuator_voxel_state[key]
        if node_voxel_input_state is not None:
            for key in (
                "voxel_coords",
                "node_features",
                "node_mask",
                "node_counts",
            ):
                value = node_voxel_input_state.get(key, None)
                if torch.is_tensor(value):
                    self.debug_tensors[f"network_voxel_node_{key}"] = value
        # Phase7-3: Node/Voxel経路のTensor系debug。
        # 長期保存用ではなく、直近forward確認用である。
        if isinstance(node_voxel_debug, dict):
            self.debug_tensors["network_voxel_node_count_tensor"] = pts_xyz.new_tensor(
                float(node_voxel_debug.get("network_voxel_node_count", 0) or 0)
            ).detach()

        # Phase2: Actuator側で作ったcanonical voxel復元debugをNetwork側にも保存する。
        # 戻り値形式は変えない。
        if isinstance(actuator_stats, dict):
            for key in (
                "canonical_voxel_coords_before",
                "canonical_voxel_coords_after",
                "voxel_restore_meta",
                "restored_xyz_debug",
                "restore_info",
            ):
                if key in actuator_stats:
                    self.debug_tensors[key] = actuator_stats[key]

        actuator_local_value = actuator_stats.get("local_recomputed", False)
        if torch.is_tensor(actuator_local_value):
            actuator_local_recomputed = bool(float(actuator_local_value.detach().float().mean().cpu()) > 0.5)
        else:
            actuator_local_recomputed = bool(actuator_local_value)
        cause_aggregation_unit_mode = ",".join(sorted(set(aggregation_unit_modes))) if aggregation_unit_modes else "unknown"
        forward_local_recomputed = bool(
            structure.get("local_recomputed", False)
            or actuator_local_recomputed
            or "local_recomputed" in cause_aggregation_unit_mode
        )
        if octree_mode_text == "prebuilt_subtree_tree" and forward_local_recomputed:
            raise RuntimeError("prebuilt_subtree_tree forward used a local_recomputed path.")

        if (
            octree_mode_text == "full_cloud"
            and bool(getattr(self.args, "full_cloud_forbid_actuator_local_recompute", True))
            and forward_local_recomputed
        ):
            raise RuntimeError(
                "full_cloud forward used a local_recomputed path. "
                f"structure_local_recomputed={bool(structure.get('local_recomputed', False))}, "
                f"actuator_local_recomputed={bool(actuator_local_recomputed)}, "
                f"cause_aggregation_unit_mode={cause_aggregation_unit_mode}, "
                f"actuator_octree_context_source={actuator_stats.get('actuator_octree_context_source', 'unknown')}, "
                f"actuator_voxel_mode={actuator_stats.get('actuator_voxel_mode', 'unknown')}"
            )
        if self._should_collect_runtime_debug() and self.writer is not None and hasattr(self.writer, "write"):
            self.writer.write(
                "NetworkStructureMode: "
                f"octree_input_mode={structure.get('octree_input_mode', octree_input_mode)}, "
                f"use_subtree_tree={bool(subtree_tree is not None)}, "
                f"use_full_octree_context={bool(full_octree_context is not None)}, "
                f"structural_voxel_mode={structure.get('structural_voxel_mode', 'unknown')}, "
                f"actuator_voxel_mode={actuator_stats.get('actuator_voxel_mode', 'unknown')}, "
                f"cause_aggregation_unit_mode={cause_aggregation_unit_mode}, "
                f"local_recomputed={forward_local_recomputed}"
                f", voxel_node_used={bool(node_voxel_debug.get('network_voxel_node_input_used', False))}, "
                f"voxel_node_fallback={bool(node_voxel_debug.get('network_voxel_node_fallback', False))}, "
                f"voxel_node_reason={node_voxel_debug.get('network_voxel_node_fallback_reason', '')}, "
                f"voxel_node_count={int(node_voxel_debug.get('network_voxel_node_count', 0) or 0)}, "
                f"voxel_node_source={node_voxel_debug.get('network_voxel_node_source', 'none')}"
            )
        soft_term_keys = (
            "add_prob_mean",
            "add_ratio",
            "add_shape_guard",
            "add_direction_ce",
            "learned_add_ratio",
            "drop_prob",
            "drop_prob_direct",
            "drop_prob_proxy",
            "drop_logit",
            "learned_drop_logit",
            "soft_drop_where_grad_base",
            "soft_drop_prob_for_ste",
            "prune_where_proxy",
            "soft_drop_where_grad_masked",
            "soft_drop_where_grad_direct",
            "drop_shape_guard",
            "learned_drop_prob",
            "learned_drop_ratio",
            "drop_prob_mean",
            "drop_prob_min",
            "drop_prob_max",
            "drop_prob_direct_mean",
            "drop_prob_direct_min",
            "drop_prob_direct_max",
            "drop_prob_proxy_mean",
            "drop_prob_proxy_min",
            "drop_prob_proxy_max",
            "drop_logit_mean",
            "drop_logit_min",
            "drop_logit_max",
            "keep_prob_mean",
            "keep_prob_min",
            "keep_prob_max",
            "drop_entropy",
            "soft_drop_mass",
            "selected_drop_count_hard",
            "prune_soft_geom",
            "prune_soft_rate",
            "prune_soft_node",
            "prune_soft_single",
            "prune_soft_bit",
            "drop_direct_target_loss",
            "keep_prob",
            "move_score_mean",
            "move_direction_ce",
            "learned_move_ratio",
            "soft_activity_loss",
            "voxel_soft_drop_score",
            "voxel_soft_add_score",
            "voxel_soft_move_score",
            "voxel_soft_drop_amount",
            "voxel_soft_add_amount",
            "voxel_soft_move_amount",
            "voxel_soft_edit_score",
            "voxel_soft_edit_count_proxy",

            "network_drop_logit_for_outcome",
            "hard_drop_mask_for_outcome",
            "hard_delete_selection_mask_for_outcome",
            "drop_ratio_hard_for_outcome",
            "raw_learned_drop_ratio_for_outcome",
            "raw_learned_drop_ratio",
            "raw_learned_add_ratio",
            "raw_learned_move_ratio",
            "drop_operation_gate",
            "add_operation_gate",
            "move_operation_gate",
            "operation_gate_logit",
            "operation_gate_prob",
            "hard_drop_target_ratio_source_id",
            "hard_drop_target_ratio_value",
            "hard_drop_target_ratio_network_value",
            "hard_drop_target_ratio_codec_prior_value",
            "amount_mode_id",
            "amount_mode_network",
            "codec_prune_prior_base_ratio",
            "codec_prune_prior_active_ratio",
            "codec_prune_prior_count_alpha",
            "learned_drop_ratio_value",
            "codec_block_valid_point_count",
            "codec_block_budget_points",
            "codec_block_count",
            "codec_block_selected_block_count",
            "codec_block_selected_point_count",
            "codec_block_budget_zero",
            "codec_block_target_drop_ratio",
            "codec_block_under_selected",
        )
        self.last_actuator_soft_terms = {
            key: value
            for key in soft_term_keys
            for value in (actuator_stats.get(key, None),)
            if torch.is_tensor(value)
        }
        try:
            setattr(self.args, "_last_actuator_soft_terms", self.last_actuator_soft_terms)
        except Exception:
            pass

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
        if isinstance(actuator_stats, dict):
            out_label = {
                "point_label": out_label,
                "canonical_voxel_coords_before": actuator_stats.get("canonical_voxel_coords_before", None),
                "canonical_voxel_coords_after": actuator_stats.get("canonical_voxel_coords_after", None),
                "voxel_restore_meta": actuator_stats.get("voxel_restore_meta", None),
                "restored_xyz_debug": actuator_stats.get("restored_xyz_debug", None),
                "restore_info": actuator_stats.get("restore_info", None),
            }
            if node_voxel_input_state is not None:
                out_label["network_voxel_node_input_used"] = True
                out_label["network_voxel_coords"] = node_voxel_input_state.get("voxel_coords", None)
                out_label["network_voxel_node_mask"] = node_voxel_input_state.get("node_mask", None)
                out_label["network_voxel_node_restore_info"] = node_voxel_input_state.get("restore_info", None)
                if bool(getattr(self.args, "voxel_node_restore_output_debug", False)):
                    final_coords_for_restore = actuator_voxel_state.get("final_voxel_coords", None)
                    final_meta_for_restore = (
                        actuator_stats.get("voxel_restore_meta", None)
                        or node_voxel_input_state.get("restore_meta", None)
                    )
                    if final_coords_for_restore is not None:
                        try:
                            restored_voxel_xyz, restored_voxel_info = restore_points_from_voxel_coords(
                                final_coords_for_restore,
                                meta=final_meta_for_restore,
                                args=self.args,
                                center=bool(getattr(self.args, "sparsepcgc_dequantize_center", False)),
                                unique=True,
                                dtype=pts_xyz.dtype,
                                device=pts_xyz.device,
                            )
                            out_label["network_voxel_node_restored_xyz_debug"] = restored_voxel_xyz
                            out_label["network_voxel_node_restored_info"] = restored_voxel_info
                        except Exception as exc:
                            out_label["network_voxel_node_restore_error"] = f"{type(exc).__name__}:{str(exc)[:160]}"

        repair_gate = actuator_stats["repair_gate"]

        def _actuator_scalar(key, default=0.0):
            value = actuator_stats.get(key, None)
            if torch.is_tensor(value):
                return float(value.detach().float().mean().cpu())
            if value is None:
                return float(default)
            try:
                return float(value)
            except Exception:
                return float(default)

        """問題スコアの算出"""
        single_chain_score = self._masked_point_mean(
            structure["single_proxy_full"].pow(2),
            selection_mask,
        )

        # Occupancy NLL proxy は、低確率ratioとは別に平均値として計算する
        occupancy_nll_score = self._masked_point_mean(
            structure["occupancy_nll_proxy_full"],
            selection_mask,
        )

        # Low-probability Occupancy proxy は、lowprob_proxy_full から計算する
        lowprob_score = self._masked_point_mean(
            structure["lowprob_proxy_full"],
            selection_mask,
        )

        # Low-probability Occupancy Ratio は、lowprob_proxy_full が閾値を超えた点の割合として計算する
        lowprob_ratio = self._masked_point_mean(
            (structure["lowprob_proxy_full"] > 0.5).to(dtype=pts_xyz.dtype),
            selection_mask,
        )

        node_score = self._masked_point_mean(
            structure["node_proxy_full"],
            selection_mask,
        )

        quant_score = self._masked_point_mean(
            structure["quant_proxy_full"],
            selection_mask,
        )

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
                structure_local_recomputed = bool(structure.get("local_recomputed", False))
                self.last_structure_debug = {
                    "actuator_octree_context_source": str(
                        actuator_stats.get("actuator_octree_context_source", "unknown")
                    ),
                    "actuator_octree_context_is_full_cloud": bool(
                        actuator_stats.get("actuator_octree_context_is_full_cloud", False)
                    ),
                    "actuator_octree_context_available": bool(
                        actuator_stats.get("actuator_octree_context_available", False)
                    ),
                    "actuator_octree_context_has_global_voxel_coords": bool(
                        actuator_stats.get("actuator_octree_context_has_global_voxel_coords", False)
                    ),
                    "actuator_voxel_state_saved": bool(
                        isinstance(getattr(self.args, "_last_actuator_voxel_state", None), dict)
                    ),
                    "actuator_final_voxel_state_available": bool(
                        isinstance(getattr(self.args, "_last_actuator_voxel_state", None), dict)
                        and getattr(self.args, "_last_actuator_voxel_state", {}).get("final_voxel_coords", None) is not None
                    ),
                    "final_voxel_update_mode": actuator_stats.get(
                        "final_voxel_update_mode",
                        "unknown",
                    ),
                    "final_voxel_recomputed_from_pts_out": bool(
                        actuator_stats.get("final_voxel_recomputed_from_pts_out", True)
                    ),
                    "voxel_edit_state_enabled": bool(actuator_stats.get("voxel_edit_state_enabled", False)),
                    "voxel_edit_mode": str(actuator_stats.get("voxel_edit_mode", "unknown")),
                    "voxel_edit_initial_count": int(actuator_stats.get("voxel_edit_initial_count", 0)),
                    "voxel_edit_final_count": int(actuator_stats.get("voxel_edit_final_count", 0)),
                    "voxel_edit_drop_count": int(actuator_stats.get("voxel_edit_drop_count", 0)),
                    "voxel_edit_add_count": int(actuator_stats.get("voxel_edit_add_count", 0)),
                    "voxel_edit_move_count": int(actuator_stats.get("voxel_edit_move_count", 0)),
                    "input_voxel_count": int(actuator_stats.get("input_voxel_count", actuator_stats.get("voxel_edit_initial_count", 0))),
                    "final_voxel_count": int(actuator_stats.get("final_voxel_count", actuator_stats.get("voxel_edit_final_count", 0))),
                    "repair_output_voxel_restored_points": bool(actuator_stats.get("repair_output_voxel_restored_points", False)),

                    "actuator_full_octree_context_available": bool(actuator_stats.get("full_octree_context_available", False)),
                    "actuator_parent_occupancy_code": int(actuator_stats.get("actuator_parent_occupancy_code", 0)),
                    "actuator_sibling_count": int(actuator_stats.get("actuator_sibling_count", 0)),
                    "actuator_ancestor_count": int(actuator_stats.get("actuator_ancestor_count", 0)),
                    "actuator_full_context_bonus_mean": float(
                        actuator_stats.get("full_context_bonus_mean", pts_xyz.new_zeros(())).detach().cpu()
                    )
                    if torch.is_tensor(actuator_stats.get("full_context_bonus_mean", None))
                    else float(actuator_stats.get("full_context_bonus_mean", 0.0)),
                    "leaf_actuator_prior_enabled": bool(
                        float(actuator_stats.get("leaf_actuator_prior_enabled", pts_xyz.new_zeros(())).detach().cpu()) > 0.5
                    )
                    if torch.is_tensor(actuator_stats.get("leaf_actuator_prior_enabled", None))
                    else bool(actuator_stats.get("leaf_actuator_prior_enabled", False)),
                    "leaf_actuator_drop_prior_mean": _actuator_scalar("leaf_actuator_drop_prior_mean"),
                    "leaf_actuator_add_prior_mean": _actuator_scalar("leaf_actuator_add_prior_mean"),
                    "leaf_actuator_move_prior_mean": _actuator_scalar("leaf_actuator_move_prior_mean"),
                    "leaf_actuator_best_prior_mean": _actuator_scalar("leaf_actuator_best_prior_mean"),
                    "leaf_actuator_best_prior_max": _actuator_scalar("leaf_actuator_best_prior_max"),
                    "octree_input_mode": str(structure.get("octree_input_mode", octree_input_mode)),
                    "use_subtree_tree": bool(subtree_tree is not None),
                    "use_full_octree_context": bool(full_octree_context is not None),
                    "structural_voxel_mode": str(structure.get("structural_voxel_mode", "unknown")),
                    "actuator_voxel_mode": str(actuator_stats.get("actuator_voxel_mode", "unknown")),
                    "cause_aggregation_unit_mode": cause_aggregation_unit_mode,
                    "local_recomputed": bool(structure_local_recomputed or actuator_local_recomputed or "local_recomputed" in cause_aggregation_unit_mode),
                    "structure_local_recomputed": structure_local_recomputed,
                    "actuator_local_recomputed": actuator_local_recomputed,
                    "prebuilt_metadata_used": bool(subtree_tree is not None and not structure_local_recomputed and not actuator_local_recomputed),
                    "prebuilt_fallback": bool((subtree_tree is not None) and (structure_local_recomputed or actuator_local_recomputed or "local_recomputed" in cause_aggregation_unit_mode)),
                    "repair_unit_keys_used": bool(
                        "prebuilt" in cause_aggregation_unit_mode
                        and (
                            full_unit_keys is not None
                            or structure.get("aggregation_unit_keys", None) is not None
                            or structure.get("structural_voxel_key", None) is not None
                            or structure.get("point_feature_voxel_key", None) is not None
                        )
                    ),
                    "aggregation_unit_key_available": bool(
                        structure.get("aggregation_unit_keys", None) is not None
                    ),
                    # ============================================================
                    # Phase4:
                    # CostAttribution / CauseAggregation / OctreeStructure が
                    # Node/Voxel基準で動いているかを確認するdebug。
                    # ============================================================
                    "phase4_cost_attribution_input_mode": str(
                        getattr(self.cost_attributor, "debug_tensors", {}).get("input_mode", "unknown")
                    ),
                    "phase4_cost_scores_requires_grad": bool(
                        getattr(self.cost_attributor, "debug_tensors", {}).get("scores_requires_grad", False)
                    ),
                    "phase4_cost_logits_requires_grad": bool(
                        getattr(self.cost_attributor, "debug_tensors", {}).get("logits_requires_grad", False)
                    ),
                    "phase4_cause_entropy": float(
                        getattr(self.cost_attributor, "debug_tensors", {}).get(
                            "cause_entropy",
                            pts_xyz.new_zeros(())
                        ).detach().cpu()
                    )
                    if torch.is_tensor(
                        getattr(self.cost_attributor, "debug_tensors", {}).get("cause_entropy", None)
                    )
                    else 0.0,
                    "phase4_aggregation_key_source": str(
                        structure.get("phase4_aggregation_key_source", "unknown")
                    ),
                    "phase4_aggregation_unit_count": int(
                        structure.get("phase4_aggregation_unit_count", 0) or 0
                    ),
                    "phase4_aggregation_min_unit_size": int(
                        structure.get("phase4_aggregation_min_unit_size", 0) or 0
                    ),
                    "phase4_aggregation_max_unit_size": int(
                        structure.get("phase4_aggregation_max_unit_size", 0) or 0
                    ),
                    "phase4_structural_key_source": str(
                        structure.get("phase4_structural_key_source", "unknown")
                    ),
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
                    "drop_operation_gate": _actuator_scalar("drop_operation_gate"),
                    "add_operation_gate": _actuator_scalar("add_operation_gate"),
                    "move_operation_gate": _actuator_scalar("move_operation_gate"),
                    "raw_learned_drop_ratio": _actuator_scalar("raw_learned_drop_ratio"),
                    "raw_learned_add_ratio": _actuator_scalar("raw_learned_add_ratio"),
                    "raw_learned_move_ratio": _actuator_scalar("raw_learned_move_ratio"),
                    "operation_gate_oracle_loss": _actuator_scalar("operation_gate_oracle_loss"),
                    "actual_oracle_candidate_where_loss": _actuator_scalar("actual_oracle_candidate_where_loss"),
                    "actual_oracle_drop_amount_loss": _actuator_scalar("actual_oracle_drop_amount_loss"),
                    "actual_oracle_add_amount_loss": _actuator_scalar("actual_oracle_add_amount_loss"),
                    "actual_oracle_move_amount_loss": _actuator_scalar("actual_oracle_move_amount_loss"),
                    "actual_oracle_drop_amount_logit_loss": _actuator_scalar("actual_oracle_drop_amount_logit_loss"),
                    "actual_oracle_add_amount_logit_loss": _actuator_scalar("actual_oracle_add_amount_logit_loss"),
                    "actual_oracle_amount_supervision_loss": _actuator_scalar("actual_oracle_amount_supervision_loss"),
                    "actual_oracle_bad_candidate_count": int(actuator_stats.get("actual_oracle_bad_candidate_count", 0)),
                    "actual_oracle_improving_candidate_count": int(actuator_stats.get("actual_oracle_improving_candidate_count", 0)),
                    "actual_oracle_combo_extra_count": int(actuator_stats.get("actual_oracle_combo_extra_count", 0)),
                    "actual_oracle_drop_bad_count": int(actuator_stats.get("actual_oracle_drop_bad_count", 0)),
                    "actual_oracle_add_bad_count": int(actuator_stats.get("actual_oracle_add_bad_count", 0)),
                    "actual_oracle_move_bad_count": int(actuator_stats.get("actual_oracle_move_bad_count", 0)),
                    "actual_oracle_drop_reason": str(actuator_stats.get("actual_oracle_drop_reason", "")),
                    "actual_oracle_operation": str(actuator_stats.get("actual_oracle_operation", "")),
                    "actual_oracle_scheduled_operation": str(
                        actuator_stats.get("actual_oracle_scheduled_operation", "")
                    ),
                    "actual_oracle_apply_teacher_actions": bool(
                        _actuator_scalar("actual_oracle_apply_teacher_actions") > 0.5
                    ),
                    "actual_gate_prune_enabled": bool(
                        _actuator_scalar("actual_gate_prune_enabled") > 0.5
                    ),
                    "actual_gate_prune_allowed": bool(
                        _actuator_scalar("actual_gate_prune_allowed") > 0.5
                    ),
                    "codec_prune_prior_enabled": bool(
                        _actuator_scalar("codec_prune_prior_enabled") > 0.5
                    ),
                    "codec_prune_prior_phase": _actuator_scalar("codec_prune_prior_phase"),
                    "codec_prune_prior_ratio": _actuator_scalar("codec_prune_prior_ratio"),
                    "codec_prune_prior_block_size": int(
                        round(_actuator_scalar("codec_prune_prior_block_size"))
                    ),
                    "codec_prune_prior_block_count_mean": _actuator_scalar(
                        "codec_prune_prior_block_count_mean"
                    ),
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
                    "repair_move_warmup": float(actuator_stats.get("repair_move_warmup", 1.0)),
                    "target_move_ratio": float(actuator_stats.get("target_move_ratio", 0.0)),
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
                    "occupancy_nll_proxy": float(occupancy_nll_score.detach().cpu()),
                    "lowprob_occupancy_score": float(lowprob_score.detach().cpu()),
                    "lowprob_occupancy_ratio": float(lowprob_ratio.detach().cpu()),
                    "single_chain_score": float(single_chain_score.detach().cpu()),
                    "node_score": float(node_score.detach().cpu()),
                    "policy_diversity": int(active_policy_count),
                    "octree_level_debug": structure.get("level_debug"),
                    "use_subtree_tree": bool(subtree_tree is not None),
                    "use_full_octree_context": bool(full_octree_context is not None),
                    "octree_input_mode": str(structure.get("octree_input_mode", "local_recomputed")),

                    # Section1:
                    # full cloud canonical voxel coordsに基づくleaf pattern診断のdebug。
                    "leaf_pattern_available": bool(structure.get("leaf_pattern_available", False)),
                    "leaf_pattern_source": str(structure.get("leaf_pattern_source", "none")),
                    "leaf_pattern_reason": str(structure.get("leaf_pattern_reason", "")),
                    "leaf_unique_parent_count": int(structure.get("leaf_unique_parent_count", 0) or 0),
                    "leaf_unique_pattern_count": int(structure.get("leaf_unique_pattern_count", 0) or 0),
                    "leaf_mean_child_count": float(structure.get("leaf_mean_child_count", 0.0) or 0.0),
                    "leaf_single_child_parent_ratio": float(structure.get("leaf_single_child_parent_ratio", 0.0) or 0.0),
                    "leaf_max_pattern_frequency": float(structure.get("leaf_max_pattern_frequency", 0.0) or 0.0),

                    # Section2/3:
                    "leaf_candidate_available": bool(structure.get("leaf_candidate_available", False)),
                    "leaf_delete_gain_mean": float(structure.get("leaf_delete_gain_mean", 0.0) or 0.0),
                    "leaf_add_gain_mean": float(structure.get("leaf_add_gain_mean", 0.0) or 0.0),
                    "leaf_move_gain_mean": float(structure.get("leaf_move_gain_mean", 0.0) or 0.0),
                    "leaf_high_gain_candidate_ratio": float(structure.get("leaf_high_gain_candidate_ratio", 0.0) or 0.0),
                    "leaf_feature_integration_used": bool(structure.get("leaf_feature_integration_used", False)),
                    "leaf_feature_best_gain_mean": float(structure.get("leaf_feature_best_gain_mean", 0.0) or 0.0),
                    "leaf_feature_best_gain_max": float(structure.get("leaf_feature_best_gain_max", 0.0) or 0.0),

                    # Section4:
                    "leaf_actuator_prior_enabled": bool(
                        _actuator_scalar("leaf_actuator_prior_enabled") > 0.5
                    ),
                    "leaf_actuator_drop_prior_mean": _actuator_scalar("leaf_actuator_drop_prior_mean"),
                    "leaf_actuator_add_prior_mean": _actuator_scalar("leaf_actuator_add_prior_mean"),
                    "leaf_actuator_move_prior_mean": _actuator_scalar("leaf_actuator_move_prior_mean"),
                    "leaf_actuator_best_prior_mean": _actuator_scalar("leaf_actuator_best_prior_mean"),
                    "leaf_actuator_best_prior_max": _actuator_scalar("leaf_actuator_best_prior_max"),

                    # Section5:
                    "leaf_target_direction_prior_enabled": bool(
                        _actuator_scalar("leaf_target_direction_prior_enabled") > 0.5
                    ),
                    "leaf_add_target_match_ratio": _actuator_scalar("leaf_add_target_match_ratio"),
                    "leaf_move_target_match_ratio": _actuator_scalar("leaf_move_target_match_ratio"),
                    "leaf_add_target_bias_mean": _actuator_scalar("leaf_add_target_bias_mean"),
                    "leaf_move_target_bias_mean": _actuator_scalar("leaf_move_target_bias_mean"),

                    "network_voxel_node_input_requested": bool(node_voxel_debug.get("network_voxel_node_input_requested", False)),
                    "leaf_actuator_prior_enabled": bool(
                        _actuator_scalar("leaf_actuator_prior_enabled") > 0.5
                    ),
                    "leaf_actuator_drop_prior_mean": _actuator_scalar("leaf_actuator_drop_prior_mean"),
                    "leaf_actuator_add_prior_mean": _actuator_scalar("leaf_actuator_add_prior_mean"),
                    "leaf_actuator_move_prior_mean": _actuator_scalar("leaf_actuator_move_prior_mean"),
                    "leaf_actuator_best_prior_mean": _actuator_scalar("leaf_actuator_best_prior_mean"),
                    "leaf_actuator_best_prior_max": _actuator_scalar("leaf_actuator_best_prior_max"),

                    "network_voxel_node_input_used": bool(node_voxel_debug.get("network_voxel_node_input_used", False)),
                    "full_cloud_anchor_node_voxel_used": bool(
                        str(octree_input_mode or "").strip().lower() == "full_cloud"
                        and bool(node_voxel_debug.get("network_voxel_node_input_used", False))
                    ),
                    "subtree_node_voxel_used": bool(
                        str(octree_input_mode or "").strip().lower() != "full_cloud"
                        and bool(node_voxel_debug.get("network_voxel_node_input_used", False))
                    ),
                    "phase7_node_voxel_debug_available": True,
                    "network_voxel_node_fallback": bool(node_voxel_debug.get("network_voxel_node_fallback", False)),
                    "network_voxel_node_fallback_reason": str(node_voxel_debug.get("network_voxel_node_fallback_reason", "")),
                    "network_voxel_node_count": int(node_voxel_debug.get("network_voxel_node_count", 0) or 0),
                    "network_voxel_node_source": str(node_voxel_debug.get("network_voxel_node_source", "none")),
                    "network_voxel_node_feature_shape": str(node_voxel_debug.get("network_voxel_node_feature_shape", "")),
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
                for key in (
                    "actual_oracle_bad_candidate_count",
                    "actual_oracle_improving_candidate_count",
                    "actual_oracle_combo_extra_count",
                    "actual_oracle_generated_candidate_count",
                    "actual_oracle_accepted_candidate_count",
                    "actual_oracle_accepted_prune_count",
                    "actual_oracle_accepted_add_count",
                    "actual_oracle_accepted_adjust_count",
                    "actual_oracle_accepted_subtree_move_count",
                    "actual_oracle_accepted_parent_collapse_count",
                    "actual_oracle_accepted_pattern_canonicalize_count",
                    "actual_oracle_noop_label_count",
                    "actual_oracle_high_rate_mppov_count",
                    "actual_oracle_low_prob_occupied_count",
                    "actual_oracle_single_child_chain_count",
                    "actual_oracle_context_pattern_candidate_count",
                    "actual_oracle_eval_count",
                    "actual_oracle_eval_max",
                ):
                    self.last_structure_debug[key] = int(round(_actuator_scalar(key, 0.0)))
                for key in (
                    "actual_oracle_noop_label_weight",
                    "actual_oracle_time",
                    "actual_oracle_edit_record_bits",
                    "actual_oracle_raw_percent",
                    "actual_oracle_delta_actual_percent",
                    "actual_oracle_proxy_percent",
                    "actual_oracle_geometry_percent",
                    "actual_oracle_original_actual_bits",
                    "actual_oracle_edited_actual_bits",
                    "operation_entropy_loss",
                    "operation_entropy_weight_effective",
                ):
                    self.last_structure_debug[key] = _actuator_scalar(key, 0.0)
                for key in (
                    "phase0_network_prune_mode",
                    "actual_oracle_force_no_edit_used",
                    "actual_oracle_has_drop",
                    "hard_prune_actual_allowed",
                    "phase0_network_mode_but_hard_drop_zero",
                    "phase0_noop_only_collapse_detected",
                    "drop_score_gate_applied_to_hard_selection",
                    "min_hard_drop_count_floor_applied",
                    "amount_mode_network",
                    "codec_block_budget_zero",
                    "codec_block_under_selected",
                ):
                    self.last_structure_debug[key] = bool(_actuator_scalar(key, 0.0) > 0.5)
                for key in (
                    "prune_after_prior_mode",
                    "hard_drop_block_reason",
                    "collapse_reason",
                    "hard_drop_count_trace",
                    "delete_candidate_empty_reason",
                ):
                    self.last_structure_debug[key] = str(actuator_stats.get(key, ""))
                for key in (
                    "codec_prune_prior_base_ratio",
                    "codec_prune_prior_active_ratio",
                    "codec_prune_prior_count_alpha",
                    "learned_drop_ratio_before_floor",
                    "learned_drop_ratio_after_floor",
                    "learned_drop_ratio_before_gate",
                    "learned_drop_ratio_after_gate",
                    "learned_drop_ratio_value",
                    "effective_drop_ratio_for_hard_count",
                    "hard_drop_target_ratio_source_id",
                    "hard_drop_target_ratio_value",
                    "hard_drop_target_ratio_network_value",
                    "hard_drop_target_ratio_codec_prior_value",
                    "amount_mode_id",
                    "network_prune_ratio_floor",
                    "network_prune_min_hard_count",
                    "network_prune_floor_steps",
                    "network_prune_floor_decay_steps",
                    "voxel_count",
                    "codec_block_valid_point_count",
                    "codec_block_budget_points",
                    "codec_block_count",
                    "codec_block_selected_block_count",
                    "codec_block_selected_point_count",
                    "codec_block_target_drop_ratio",
                    "delete_candidate_count",
                    "delete_candidate_point_count",
                    "hard_delete_selection_count",
                    "pre_round_target_count",
                    "post_round_target_count",
                    "hard_mask_count",
                    "final_hard_drop_count",
                    "selected_drop_count_hard",
                ):
                    self.last_structure_debug[key] = _actuator_scalar(key, 0.0)
        else:
            self.last_structure_debug = {
                "network_voxel_node_input_requested": bool(node_voxel_debug.get("network_voxel_node_input_requested", False)),
                "network_voxel_node_input_used": bool(node_voxel_debug.get("network_voxel_node_input_used", False)),
                "network_voxel_node_fallback": bool(node_voxel_debug.get("network_voxel_node_fallback", False)),
                "network_voxel_node_fallback_reason": str(node_voxel_debug.get("network_voxel_node_fallback_reason", "")),
                "network_voxel_node_count": int(node_voxel_debug.get("network_voxel_node_count", 0) or 0),
                "network_voxel_node_source": str(node_voxel_debug.get("network_voxel_node_source", "none")),
                "network_voxel_node_feature_shape": str(node_voxel_debug.get("network_voxel_node_feature_shape", "")),
                "phase4_cost_attribution_input_mode": str(
                    getattr(self.cost_attributor, "debug_tensors", {}).get("input_mode", "unknown")
                ),
                "phase4_aggregation_key_source": str(
                    structure.get("phase4_aggregation_key_source", "unknown")
                    if isinstance(structure, dict)
                    else "unknown"
                ),
                "phase4_aggregation_unit_count": int(
                    structure.get("phase4_aggregation_unit_count", 0)
                    if isinstance(structure, dict)
                    else 0
                ),
                "phase4_structural_key_source": str(
                    structure.get("phase4_structural_key_source", "unknown")
                    if isinstance(structure, dict)
                    else "unknown"
                ),

                # Section1:
                # full cloud canonical voxel coordsに基づくleaf pattern診断のdebug。
                "leaf_pattern_available": bool(
                    structure.get("leaf_pattern_available", False)
                    if isinstance(structure, dict)
                    else False
                ),
                "leaf_pattern_source": str(
                    structure.get("leaf_pattern_source", "none")
                    if isinstance(structure, dict)
                    else "none"
                ),
                "leaf_pattern_reason": str(
                    structure.get("leaf_pattern_reason", "")
                    if isinstance(structure, dict)
                    else "missing"
                ),
                "leaf_unique_parent_count": int(
                    structure.get("leaf_unique_parent_count", 0)
                    if isinstance(structure, dict)
                    else 0
                ),
                "leaf_unique_pattern_count": int(
                    structure.get("leaf_unique_pattern_count", 0)
                    if isinstance(structure, dict)
                    else 0
                ),
                "leaf_mean_child_count": float(
                    structure.get("leaf_mean_child_count", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_single_child_parent_ratio": float(
                    structure.get("leaf_single_child_parent_ratio", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_max_pattern_frequency": float(
                    structure.get("leaf_max_pattern_frequency", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_candidate_available": bool(
                    structure.get("leaf_candidate_available", False)
                    if isinstance(structure, dict)
                    else False
                ),
                "leaf_delete_gain_mean": float(
                    structure.get("leaf_delete_gain_mean", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_add_gain_mean": float(
                    structure.get("leaf_add_gain_mean", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_move_gain_mean": float(
                    structure.get("leaf_move_gain_mean", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_high_gain_candidate_ratio": float(
                    structure.get("leaf_high_gain_candidate_ratio", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_feature_integration_used": bool(
                    structure.get("leaf_feature_integration_used", False)
                    if isinstance(structure, dict)
                    else False
                ),
                "leaf_feature_best_gain_mean": float(
                    structure.get("leaf_feature_best_gain_mean", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
                "leaf_feature_best_gain_max": float(
                    structure.get("leaf_feature_best_gain_max", 0.0)
                    if isinstance(structure, dict)
                    else 0.0
                ),
            }

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
