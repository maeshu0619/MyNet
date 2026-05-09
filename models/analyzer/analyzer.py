import time
import torch
import torch.nn as nn
from ..utils.pointcloud.utils_repkpu import get_knn_pts
import torch.nn.functional as Fnn
from ..utils.compression.proxy_octree import ProxyOctreeConfig, SoftOctreeRateProxy

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
        self.scale = float(getattr(self.cfgs, "outlier_label_th_scale", 4.0)) # 外れ点ラベルの閾値倍率
        self.outlier_min_ratio = float(getattr(self.cfgs, "outlier_label_min_ratio", 0.03))
        self.outlier_max_ratio = float(getattr(self.cfgs, "outlier_label_max_ratio", 0.15))

        proxy_cfg = ProxyOctreeConfig(
            max_depth=int(getattr(cfgs, "proxy_max_depth", 12)),
            qs=float(getattr(cfgs, "qs", 2.0)),
            round_tau=float(getattr(cfgs, "proxy_round_tau", 0.12)),
            mass_to_occ_gain=float(getattr(cfgs, "proxy_mass_to_occ_gain", 1.0)),
            ctx_dim=self.octree_ctx_dim,
        )
        self.proxy_octree_ctx = SoftOctreeRateProxy(proxy_cfg)

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
        fft_norm = torch.abs(nn_fft).pow(2).sum(dim=-1, keepdim=True).sqrt()  # [B,N,1]
        fft_norm = fft_norm.real.to(dtype=dist.dtype)

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
        outlier_score = outlier_score / (outlier_score.mean(dim=2, keepdim=True) + eps)
        outlier_score = outlier_score.real.to(dtype=pts.dtype)
        return outlier_score

    def _build_ssr_outlier_label(self, out_score):
        eps = 1e-6

        # 複素数が来た場合でも中央値計算できるように実数化する
        if torch.is_complex(out_score):
            out_score = out_score.abs()
        out_score = out_score.to(dtype=torch.float32)

        med = out_score.median(dim=2, keepdim=True).values
        mad = (out_score - med).abs().median(dim=2, keepdim=True).values

        mad = mad.clamp_min(eps)
        th = med + self.scale * mad
        out_label = (out_score >= th).float()

        # MAD threshold alone can become too conservative on smooth video frames.
        # Keep the label density in a small, bounded top-score band so Pruning
        # receives a stable "drop these" signal without deleting too much.
        min_ratio = max(float(self.outlier_min_ratio), 0.0)
        max_ratio = min(max(float(self.outlier_max_ratio), min_ratio), 1.0)
        if min_ratio > 0.0 or max_ratio < 1.0:
            B, _, N = out_score.shape
            score_2d = out_score.squeeze(1)
            label_2d = out_label.squeeze(1)
            min_k = int(round(min_ratio * N))
            max_k = int(round(max_ratio * N))
            min_k = max(0, min(min_k, N))
            max_k = max(min_k, min(max_k, N))
            for b in range(B):
                count = int(label_2d[b].sum().detach().item())
                if max_k > 0 and count > max_k:
                    idx = torch.topk(score_2d[b], k=max_k, largest=True).indices
                    fixed = torch.zeros_like(label_2d[b])
                    fixed.scatter_(0, idx, 1.0)
                    label_2d[b] = fixed
                elif count < min_k:
                    idx = torch.topk(score_2d[b], k=min_k, largest=True).indices
                    fixed = torch.zeros_like(label_2d[b])
                    fixed.scatter_(0, idx, 1.0)
                    label_2d[b] = fixed
            out_label = label_2d.unsqueeze(1)
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

    def _build_octree_context(self, pts, coord_scale=None): # 各点に対して軽量なOctree文脈特徴を作る関数
        qs_override = None
        if coord_scale is not None:
            if torch.is_tensor(coord_scale):
                qs_override = float(getattr(self.cfgs, "qs", 2.0)) / coord_scale.reshape(-1).clamp_min(1e-9)
            else:
                qs_override = float(getattr(self.cfgs, "qs", 2.0)) / max(float(coord_scale), 1e-9)
        with torch.no_grad():
            oct_score = self.proxy_octree_ctx.build_point_context(
                pts_xyz=pts,
                ctx_level=self.octree_ctx_level,
                final_w=None,
                qs_override=qs_override,
            )
        return oct_score

    def forward(self, pts, F, need_outlier=True, coord_scale=None):
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

        oct_score = self._build_octree_context(pts, coord_scale=coord_scale) # Octree文脈特徴の構築

        if need_outlier:
            out_score = self._build_ssr_outlier_score(pts, knn_pts, dist) # 外れ点スコアの構築
            out_label = self._build_ssr_outlier_label(out_score) # 外れ点ラベルの構築
        else:
            out_score = None
            out_label = None

        return density_score, structure_score, oct_score, out_score, out_label
