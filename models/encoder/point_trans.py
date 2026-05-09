import torch
import torch.nn as nn
import math
from models.utils.pointcloud.utils_repkpu import get_knn_pts, index_points
from torch.nn.functional import gumbel_softmax

# RepKPUから流用

class Encoder_Attention(nn.Module):
    def __init__(self, cfgs, geo_dim=3):
        super().__init__()
        self.k = cfgs.k
        
        self.q_conv = nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1)
        self.k_conv = nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1)
        self.v_conv = nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1)
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(geo_dim, cfgs.encoder_dim//2, 1, 1),
            nn.BatchNorm2d(cfgs.encoder_dim//2) if cfgs.encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfgs.encoder_dim//2, cfgs.encoder_dim, 1, 1)
        )
        self.rel_mlp = nn.Sequential(
            nn.Conv2d(cfgs.encoder_dim, cfgs.encoder_dim//2, 1, 1),
            nn.BatchNorm2d(cfgs.encoder_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfgs.encoder_dim//2, cfgs.encoder_dim, 1, 1)
        )
        # self.attn_conv = nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1)
        self.out_conv = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim*2, 1, 1),
            nn.BatchNorm1d(cfgs.encoder_dim*2) if cfgs.encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfgs.encoder_dim*2, cfgs.encoder_dim, 1, 1),
            nn.BatchNorm1d(cfgs.encoder_dim) if cfgs.encoder_bn else nn.Identity()
        )
        self.query_chunk_size = max(int(getattr(cfgs, "encoder_query_chunk", 0)), 0)
        # self.out_act = nn.ReLU(inplace=True)

    def forward(self, pts, feats, geos=None, knn_idx=None):
        
        if geos == None:
            geos = pts
        
        q = self.q_conv(feats)
        k = self.k_conv(feats)
        v = self.v_conv(feats)

        if knn_idx is None:
            _, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)

        query_chunk = self.query_chunk_size
        num_points = pts.shape[-1]
        if query_chunk <= 0 or query_chunk >= num_points:
            query_chunk = num_points

        agg_chunks = []
        for start in range(0, num_points, query_chunk):
            end = min(start + query_chunk, num_points)
            knn_idx_chunk = knn_idx[:, start:end, :]
            q_chunk = q[:, :, start:end]
            feats_chunk = feats[:, :, start:end]
            geos_chunk = geos[:, :, start:end]

            knn_geos = index_points(geos, knn_idx_chunk)
            geo_embedding = self.geo_mlp(geos_chunk.unsqueeze(-1) - knn_geos)
            knn_k = index_points(k, knn_idx_chunk)
            knn_v = index_points(v, knn_idx_chunk)

            attn = torch.softmax(self.rel_mlp(q_chunk.unsqueeze(-1) - knn_k + geo_embedding), dim=-1)
            agg_chunks.append(torch.einsum('bcnk, bcnk -> bcn', attn, knn_v + geo_embedding) + feats_chunk)

        agg_feat = torch.cat(agg_chunks, dim=2)
        out_feat = self.out_conv(agg_feat) + agg_feat
        return out_feat

class PointTransformer(nn.Module):
    def __init__(self, cfgs, input_dim=3):
        super().__init__()
        encoder_input_dim = int(getattr(cfgs, "encoder_input_dim", input_dim))
        self.stem = nn.Sequential(
            nn.Conv1d(encoder_input_dim, cfgs.encoder_dim, 1),
            nn.BatchNorm1d(cfgs.encoder_dim) if cfgs.encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1),
            nn.BatchNorm1d(cfgs.encoder_dim) if cfgs.encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        geo_dim = 3
        self.attn11 = Encoder_Attention(cfgs, geo_dim)
        self.attn12 = Encoder_Attention(cfgs, geo_dim)
        self.attn13 = Encoder_Attention(cfgs, geo_dim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim*3, cfgs.encoder_dim, 1),
            nn.BatchNorm1d(cfgs.encoder_dim),
            nn.ReLU(inplace=True)
        )

        self.attn21 = Encoder_Attention(cfgs, geo_dim)
        self.attn22 = Encoder_Attention(cfgs, geo_dim)
        self.attn23 = Encoder_Attention(cfgs, geo_dim)
        self.conv2 = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim*3, cfgs.encoder_dim, 1),
            nn.BatchNorm1d(cfgs.encoder_dim),
            nn.ReLU(inplace=True)
        )


        self.attn31 = Encoder_Attention(cfgs, geo_dim)
        self.attn32 = Encoder_Attention(cfgs, geo_dim)
        self.attn33 = Encoder_Attention(cfgs, geo_dim)
        self.conv3 = nn.Sequential(
            nn.Conv1d(cfgs.encoder_dim*3, cfgs.encoder_dim, 1),
            nn.BatchNorm1d(cfgs.encoder_dim),
            nn.ReLU(inplace=True)
        )
        if cfgs.global_mlp:
            self.global_mlp = nn.Sequential(
                nn.Conv1d(cfgs.encoder_dim, 64, 1),
                nn.BatchNorm1d(64) if cfgs.encoder_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128) if cfgs.encoder_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.fuse_mlp = nn.Sequential(
                nn.Conv1d(cfgs.encoder_dim*4+128, cfgs.out_dim*2, 1),
                nn.BatchNorm1d(cfgs.out_dim*2),
                nn.ReLU(inplace=True),
                nn.Conv1d(cfgs.out_dim*2, cfgs.out_dim, 1),
                nn.BatchNorm1d(cfgs.out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.global_mlp = nn.Identity()
            self.fuse_mlp = nn.Sequential(
                    nn.Conv1d(cfgs.encoder_dim*5, cfgs.out_dim, 1),
                    nn.BatchNorm1d(cfgs.out_dim),
                    nn.ReLU(inplace=True)
                )


    def forward(self, pts, feat=None):
        if feat == None:
            feat = pts
        feat0 = self.stem(feat)
        _, knn_idx = get_knn_pts(self.attn11.k, pts, pts, return_idx=True)

        feat11 = self.attn11(pts, feat0, None, knn_idx=knn_idx)
        feat12 = self.attn12(pts, feat11, None, knn_idx=knn_idx)
        feat13 = self.attn13(pts, feat12, None, knn_idx=knn_idx)
        feat1 = self.conv1(torch.cat([feat11, feat12, feat13], dim=1))

        feat21 = self.attn21(pts, feat1, None, knn_idx=knn_idx)
        feat22 = self.attn22(pts, feat21, None, knn_idx=knn_idx)
        feat23 = self.attn23(pts, feat22, None, knn_idx=knn_idx)
        feat2 = self.conv2(torch.cat([feat21, feat22, feat23], dim=1))

        feat31 = self.attn31(pts, feat2, None, knn_idx=knn_idx)
        feat32 = self.attn32(pts, feat31, None, knn_idx=knn_idx)
        feat33 = self.attn33(pts, feat32, None, knn_idx=knn_idx)
        feat3 = self.conv3(torch.cat([feat31, feat32, feat33], dim=1))

        global_feat = torch.max(self.global_mlp(feat3), dim=-1, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, pts.shape[-1])

        # 局所特徴（点ごと）
        F_local = torch.cat([feat1, feat2, feat3], dim=1)

        # 大域特徴（シーン全体）
        # F_global = torch.max(self.global_mlp(feat3), dim=-1)[0]

        # 特徴の統合
        F_fused =  self.fuse_mlp(torch.cat([feat0, feat1, feat2, feat3, global_feat], dim=1))

        return F_local, F_fused
