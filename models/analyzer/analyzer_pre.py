import time
import torch
import torch.nn as nn
from ..utils.pointcloud.utils_repkpu import get_knn_pts
import torch.nn.functional as Fnn

"""
Density & Structure Analyzer Module

入力:
    pts      : [B, 3, N]        点座標
    F    : [B, C, N]        Encoderの局所特徴（F_local）

出力:
    density_score   : [B, 1, N] 密度スコア d
    structure_score : [B, 1, N] 構造スコア s
"""

class DensityStructureAnalyzer(nn.Module):
    def __init__(self, cfgs, writer):
        super().__init__()
        self.k = cfgs.k
        self.writer = writer
        self.cfgs = cfgs

        self.octree_qlevel = int(getattr(cfgs, "octree_qlevel", 12)) # Octree量子化レベル
        self.octree_ctx_level = int(getattr(cfgs, "octree_ctx_level", 5)) # Octree文脈を取る階層レベル
        self.octree_ctx_dim = int(getattr(cfgs, "octree_ctx_dim", 14)) # Octree文脈次元数
        # self.th_scale = float(getattr(self.cfgs, "outlier_label_th_scale", 1.0)) # 外れ点ラベルの閾値倍率
        self.scale = float(getattr(self.cfgs, "outlier_label_th_scale", 6.0)) # 外れ点ラベルの閾値倍率

    def _build_ssr_outlier_score(self, pts, knn_pts, dist):
        # pts     : [B,3,N]
        # knn_pts : [B,3,N,K]
        # dist    : [B,N,K]

        eps = 1e-6

        # ===== 1. 近傍距離の基本統計 =====
        dist_mean = dist.mean(dim=-1, keepdim=True)                 # [B,N,1]
        dist_var  = dist.var(dim=-1, keepdim=True, unbiased=False)  # [B,N,1]
        dist_std  = torch.sqrt(dist_var + eps)                      # [B,N,1]

        # ===== 2. FFTベースの異常度 =====
        nn_signal = dist + 1.0                                      # [B,N,K]
        nn_fft = torch.fft.fft(nn_signal, dim=-1, norm='backward')
        fft_norm = nn_fft.norm(p=2, dim=-1, keepdim=True)           # [B,N,1]

        # ===== 3. バッチ内で各指標を正規化 =====
        def _normalize_feature(x):
            # x : [B,N,1]
            x = x.permute(0, 2, 1).contiguous()                     # [B,1,N]
            x = x / (x.mean(dim=2, keepdim=True) + eps)             # 平均1に正規化
            return x

        mean_score = _normalize_feature(dist_mean)                  # [B,1,N]
        std_score  = _normalize_feature(dist_std)                   # [B,1,N]
        fft_score  = _normalize_feature(fft_norm)                   # [B,1,N]

        # ===== 4. 総合外れ点スコア =====
        # 周囲から離れている点を強めに見るため mean を最重要視
        outlier_score = (
            0.5 * mean_score +
            0.2 * std_score +
            0.3 * fft_score
        )

        # 最後にもう一度平均1へ揃える
        outlier_score = outlier_score / (outlier_score.mean(dim=2, keepdim=True) + eps)

        return outlier_score

    def _build_ssr_outlier_label(self, out_score):
        # out_score : [B,1,N]
        eps = 1e-6

        # ロバスト閾値: median + scale * MAD
        med = out_score.median(dim=2, keepdim=True).values
        mad = (out_score - med).abs().median(dim=2, keepdim=True).values

        # MADが極端に小さいときのゼロ割れ防止
        mad = mad.clamp_min(eps)

        th = med + self.scale * mad

        # 1 = outlier, 0 = inlier
        out_label = (out_score >= th).float()
        return out_label
            
    # def _build_ssr_outlier_score(self, pts, knn_pts, dist): # 外れ点スコアを作る関数
    #     eps = 1e-6
    #     nn_signal = dist + 1.0 # [B,N,K] # 近傍距離distに1.0を足してFFTにいれる信号を作る
    #     nn_fft = torch.fft.fft(nn_signal, dim=-1, norm='backward') # 各点毎の近傍距離系列にFFTを掛ける
    #     norm_fft = nn_fft.norm(p=2, dim=-1, keepdim=True) # [B,N,1] # FFT結果の大きさをL2ノルムで1つの値にまとめる

    #     outlier_score = norm_fft.permute(0, 2, 1).contiguous() # [B,1,N] # 形を変換
    #     outlier_score = outlier_score / (outlier_score.mean(dim=2, keepdim=True) + eps) # 平均1になるように正規化        
    #     return outlier_score
    
    # def _build_ssr_outlier_label(self, out_score): # 外れ点スコアから外れ点ラベルを作る関数
    #     th = out_score.mean(dim=2, keepdim=True) * self.th_scale # 各バッチの平均スコアに倍率を掛けて閾値を作る
    #     out_label = (out_score >= th).float() # 閾値以上の点を1、未満を0にして外れ点ラベルを作る
    #     return out_label

    def _build_octree_context(self, pts): # 各点に対して軽量なOctree文脈特徴を作る関数
        B, _, _ = pts.shape # バッチサイズの取得
        eps = 1e-6

        with torch.no_grad(): # この中の計算では勾配を取らない
            p_min = pts.min(dim=2, keepdim=True).values # 各座標軸の最小値
            p_max = pts.max(dim=2, keepdim=True).values # 各座標軸の最大値
            p_rng = (p_max - p_min).clamp_min(eps) # 座標範囲
            pts01 = (pts - p_min) / p_rng # 点座標を正規化
            pts01 = pts01.clamp(0.0, 1.0 - 1e-6) # 数値誤差で1を超えないようにする

            qmax = (1 << self.octree_qlevel) - 1 # 量子化で使う最大整数値
            qxyz = torch.floor(pts01 * qmax).long() # [B,3,N] # 正規化座標を整数格子へ量子化

            l = self.octree_ctx_level # 文脈を取るOctreeレベルを取り出す
            shift = max(self.octree_qlevel - l, 0) # 上位レベルへ合わせるためのbit shift量を決める

            vox = torch.bitwise_right_shift(qxyz, shift) # [B,3,N] # 点を対象レベルのVoxel座標へ変換
            parent_vox = torch.bitwise_right_shift(vox, 1) # [B,3,N] # 親Voxel座標を作る

            child_bits = vox & 1 # 各軸に関してVoxelの下位bitを取り出す
            child_id = ( # xyzの下位bitをまとめて0～7のIDにする
                (child_bits[:, 0, :] << 2) |
                (child_bits[:, 1, :] << 1) |
                child_bits[:, 2, :]
            ) # [B,N], 0..7
            vox_f = vox.float() # Voxel座標をfloatに変換
            local = (pts01 * float(2 ** l) - vox_f).clamp(0.0, 1.0) # [B,3,N] # 各点の相対位置の計算

            ctx_list = []
            grid_size = 1 << max(l - 1, 0) # 親Voxel座標を1本の整数に変換するためのグリッドサイズの決定
            for b in range(B):
                # 親Voxelの座標を取り出す
                px = parent_vox[b, 0] # [N]
                py = parent_vox[b, 1]
                pz = parent_vox[b, 2]
                cid = child_id[b] # [N] # Child IDを取り出す
                
                parent_key = px * (grid_size * grid_size) + py * grid_size + pz # [N] # 親voxelを一意な1本の整数keyへ変換
                uniq_key, inverse = torch.unique(parent_key, sorted=True, return_inverse=True) # 同じ親voxelをまとめるために、一意キーと逆indexを作る
                child_oh = torch.nn.functional.one_hot(cid, num_classes=8).float() # [N,8] # 各点のchildをone-hotベクトルにする
                occ_count = torch.zeros((uniq_key.shape[0], 8), device=pts.device, dtype=torch.float32) # 親voxelごとにchild occupancyを集計
                occ_count.index_add_(0, inverse, child_oh) # 同じ親VoxelごとにChildの出現回数を足し合わせる
                occ_bin = (occ_count > 0).float() # [M,8] # Childが存在するか否かを0/1にする

                popcount_parent = occ_bin.sum(dim=1) / 8.0 # [M] # 各親Voxelの占有Childの割合計算
                is_single_parent = (occ_bin.sum(dim=1) == 1).float() # [M] # 親Voxelの単一子ノード数
                popcount = popcount_parent[inverse] # [N] # 親VoxelごとのPopcountを各点へ戻す
                is_single = is_single_parent[inverse] # [N] # 親Voxelごとの単一子ノード判定を各点に戻す
                feat_b = torch.cat([ # 各点に対するOctree特徴
                    popcount.unsqueeze(0),   # [1,N] # popcount
                    is_single.unsqueeze(0),  # [1,N] # 単一子ノード
                    local[b],                # [3,N] # 相対位置
                ], dim=0)                    # [5,N]
                ctx_list.append(feat_b) # Octree特徴をリストへ追加
            oct_score = torch.stack(ctx_list, dim=0) # [B,5,N] # バッチをまとめる
        return oct_score

    def forward(self, pts, F):
        eps = 1e-6
        
        # ===== kNN取得 =====
        knn_pts, _ = get_knn_pts(self.k, pts, pts, return_idx=True) # kNN近傍点の取得

        # ===== 距離計算 =====
        center = pts.unsqueeze(-1) # 中心点を近傍方向にbroaadcatできる形へ変換
        diff = knn_pts - center # 近傍点と中心点の差分ベクトルを計算
        dist = torch.norm(diff, dim=1) # 中心点から近傍点までの距離を計算

        # ===== 近傍統計量 =====
        dist_mean = dist.mean(dim=-1, keepdim=True) # 各点の近傍距離の平均を計算
        dist_var  = dist.var(dim=-1, keepdim=True) # 各点の近傍距離の分散を計算

        density = 1.0 / (dist_mean + eps) # 平均距離が小さいほど密度が高い都見なし、密度スコアを算出
        structure = dist_var / (dist_mean ** 2 + eps) # 近傍距離のばらつきを平均距離で正規化し、構造スコアを算出
        density_score = density.permute(0, 2, 1).contiguous() # 整形
        structure_score = structure.permute(0, 2, 1).contiguous() # 整形
        density_score = density_score / (density_score.mean(dim=2, keepdim=True) + eps) # 平均1になるように正規化
        structure_score = structure_score / (structure_score.mean(dim=2, keepdim=True) + eps) # 平均1になるように正規化

        oct_score = self._build_octree_context(pts) # Octree文脈特徴の構築        
        out_score = self._build_ssr_outlier_score(pts, knn_pts, dist) # 外れ点スコアの構築
        out_label = self._build_ssr_outlier_label(out_score) # 外れ点ラベルの構築

        return density_score, structure_score, oct_score, out_score, out_label