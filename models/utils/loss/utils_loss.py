import subprocess
import os
import json
import torch
import time
import sys
import numpy as np
import hashlib

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from models.utils.pointcloud.utils_p2c import *
except ImportError:
    pass

# from pytorch3d.ops.points_normals import estimate_pointcloud_normals

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.append(ROOT_DIR)

from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()



def estimate_normals_open3d(gt_pts: torch.Tensor, k: int = 16) -> torch.Tensor:
    if o3d is None:
        return estimate_normals_pca(gt_pts, k=k)

    # Open3DのKDTreeで法線を1回で計算（C++実装で速い）
    pts = gt_pts[0].detach().transpose(0, 1).cpu().numpy().astype(np.float64, copy=False)  # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    n = np.asarray(pcd.normals).astype(np.float32, copy=False)  # (N,3)
    normals = torch.from_numpy(n).to(gt_pts.device).transpose(0, 1).unsqueeze(0).contiguous()  # [1,3,N]
    return normals

def estimate_normals_pca(gt_pts: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    gt_pts : [B, 3, N]
    return : normals [B, 3, N]
    方針  : FAISSで各点のkNNを取り、PCAで法線を算出する。cdistは禁止。
    """
    assert gt_pts.ndim == 3
    B, _, N = gt_pts.shape
    if N <= 1:
        return torch.zeros_like(gt_pts)
    k = max(1, min(int(k), N - 1))

    # [B, N, 3]
    xyz = gt_pts.permute(0, 2, 1).contiguous()

    # 各点のkNN（自身は除外する）
    knn_idx = _torch_knn_idx(xyz, xyz, k, exclude_self=True)  # [B, N, k]

    # 近傍点取得: [B, N, k, 3]
    # knn_pts = torch.gather(
    #     xyz.unsqueeze(2).expand(-1, -1, N, -1),  # これは巨大になるので不可
    # )

    # reshape を使って安全に近傍点を集める
    # xyz_flat: [B*N, 3]
    xyz_flat = xyz.reshape(B * N, 3)

    # knn_idx_flat: [B*N*k]
    base = (torch.arange(B, device=xyz.device).view(B, 1, 1) * N)  # [B,1,1]
    knn_idx_flat = (knn_idx + base).reshape(-1)                    # [B*N*k]

    knn_pts = xyz_flat[knn_idx_flat].reshape(B, N, k, 3)           # [B,N,k,3]

    centroid = knn_pts.mean(dim=2, keepdim=True)                   # [B,N,1,3]
    diff = knn_pts - centroid                                      # [B,N,k,3]
    cov = diff.transpose(-1, -2) @ diff                            # [B,N,3,3]

    eigvals, eigvecs = torch.linalg.eigh(cov)                      # [B,N,3], [B,N,3,3]
    normals = eigvecs[..., 0]                                      # 最小固有値の固有ベクトル [B,N,3]
    normals = torch.nn.functional.normalize(normals, dim=-1)

    return normals.permute(0, 2, 1).contiguous()                   # [B,3,N]



def _torch_knn_idx(query_xyz_bnm: torch.Tensor, ref_xyz_brm: torch.Tensor, k: int,
                   exclude_self: bool = False, q_chunk: int = 2048) -> torch.Tensor:
    """
    query_xyz_bnm: [B, N, 3]
    ref_xyz_brm  : [B, M, 3]
    return       : [B, N, k]
    """
    B, N, _ = query_xyz_bnm.shape
    _, M, _ = ref_xyz_brm.shape
    if M == 0:
        raise ValueError("ref_xyz_brm must contain at least one point")

    k_max = M - 1 if exclude_self and M > 1 else M
    k = max(1, min(int(k), k_max))
    idx_list = []
    same_storage = exclude_self and N == M and query_xyz_bnm.data_ptr() == ref_xyz_brm.data_ptr()

    for b in range(B):
        batch_idx = []
        for qs in range(0, N, q_chunk):
            qe = min(qs + q_chunk, N)
            dist = torch.cdist(
                query_xyz_bnm[b:b + 1, qs:qe],
                ref_xyz_brm[b:b + 1],
            ).squeeze(0)

            if same_storage:
                row_idx = torch.arange(qs, qe, device=dist.device)
                dist.scatter_(1, row_idx.unsqueeze(1), float("inf"))

            batch_idx.append(dist.topk(k=k, largest=False).indices)

        idx_list.append(torch.cat(batch_idx, dim=0))

    return torch.stack(idx_list, dim=0)

def _faiss_knn_idx(query_xyz_bnm: torch.Tensor, ref_xyz_brm: torch.Tensor, k: int, 
                   cpu_index = None, gpu_index = None) -> torch.Tensor:
    """
    query_xyz_bnm: [B, N, 3]  最近傍を探したい点群
    ref_xyz_brm  : [B, M, 3]  参照点群（ここから最近傍を取る）
    return       : [B, N, k]  参照点群側のインデックス（int64）
    注意: FAISSはCPU numpyを要求するのでref/queryはdetach->cpuに落として検索する。
          インデックスはGPUに戻して gather に使う。
    """
    if faiss is None:
        return _torch_knn_idx(query_xyz_bnm, ref_xyz_brm, k)

    res = faiss.StandardGpuResources()
    
    B, N, _ = query_xyz_bnm.shape
    _, M, _ = ref_xyz_brm.shape
    idx_list = []

    t0 = time.time()
    for b in range(B):
        ref_np = ref_xyz_brm[b].detach().cpu().numpy().astype(np.float32, copy=False)
        qry_np = query_xyz_bnm[b].detach().cpu().numpy().astype(np.float32, copy=False)

        cpu_index = faiss.IndexFlatL2(3)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(ref_np)
        _, I = gpu_index.search(qry_np, k)
        idx_list.append(torch.from_numpy(I).to(ref_xyz_brm.device, dtype=torch.long))

    return torch.stack(idx_list, dim=0)  # [B, N, k]




def remove_outlier_points_by_label(gt_pts: torch.Tensor, outlier_label: torch.Tensor):
    """
    外れ点ラベルに基づいて GT 点群から外れ点を除去する関数

    引数:
        gt_pts        : [B, C, N]
        outlier_label : [B, 1, N] または [B, N]
                        1 = 外れ点, 0 = 内点

    戻り値:
        gt_inlier : [B, C, Ni]
                    外れ点を除去した点群
                    現時点では B=1 を前提とする
    """
    if gt_pts.dim() != 3:
        raise ValueError(f"gt_pts は [B, C, N] である必要があるが、実際は {gt_pts.shape}")

    if outlier_label.dim() == 3:
        if outlier_label.size(1) != 1:
            raise ValueError(
                f"outlier_label が3次元の場合は [B,1,N] を想定しているが、実際は {outlier_label.shape}"
            )
        label = outlier_label.squeeze(1)  # [B, N]
    elif outlier_label.dim() == 2:
        label = outlier_label
    else:
        raise ValueError(
            f"outlier_label は [B,1,N] または [B,N] である必要があるが、実際は {outlier_label.shape}"
        )

    if gt_pts.size(0) != label.size(0) or gt_pts.size(2) != label.size(1):
        raise ValueError(
            f"gt_pts.shape={gt_pts.shape} と outlier_label.shape={outlier_label.shape} の B または N が一致していない"
        )

    # 現状は B=1 前提
    if gt_pts.size(0) != 1:
        raise ValueError(
            f"この関数は現時点では B=1 前提であるが、実際の B は {gt_pts.size(0)}"
        )

    # 1 = outlier, 0 = inlier
    inlier_mask = (label[0] < 0.5)   # [N]
    gt_inlier = gt_pts[0, :, inlier_mask].unsqueeze(0)  # [1, C, Ni]

    return gt_inlier

def chamfer_l2_loss(gen_pts: torch.Tensor, gt_pts: torch.Tensor, final_w: torch.Tensor = None) -> torch.Tensor:
    """
    gen_pts, gt_pts: [B,3,N]
    """
    if final_w is None:
        # RepKPU と同様に正規化
        gen = gen_pts.transpose(1, 2).contiguous()
        gt  = gt_pts.transpose(1, 2).contiguous()

        dist1, dist2, _, _ = chamfer_dist(gen, gt)
        return dist1.mean() + dist2.mean()
    elif final_w is not None:
        gen = gen_pts.transpose(1, 2).contiguous()  # [B,N,3]
        gt  = gt_pts.transpose(1, 2).contiguous()   # [B,M,3]

        dist1, dist2, _, _ = chamfer_dist(gen, gt)  # dist1: [B,N], dist2: [B,M]

        if final_w.dim() == 3:
            w = final_w.squeeze(1)  # [B,N]
        else:
            w = final_w
        w = w.clamp(0.0, 1.0).to(dist1.dtype)

        # gen->gt を重み付き平均、gt->gen は従来通り平均（ここは最小実装）
        loss1 = (dist1 * w).sum() / (w.sum() + 1e-12)
        loss2 = dist2.mean()
        return loss1 + loss2


def chamfer_l2_loss_and_weight_surrogate(
    gen_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    final_w: torch.Tensor,
):
    """
    Return hard Chamfer and a weight-only STE surrogate using one NN pass.
    The hard loss keeps the normal point-position gradient; the surrogate
    reuses detached distances so gradients only flow to final_w.
    """
    gen = gen_pts.transpose(1, 2).contiguous()
    gt = gt_pts.transpose(1, 2).contiguous()
    dist1, dist2, _, _ = chamfer_dist(gen, gt)

    hard_loss = dist1.mean() + dist2.mean()

    if final_w.dim() == 3:
        w = final_w.squeeze(1)
    else:
        w = final_w
    w = w.clamp(0.0, 1.0).to(dist1.dtype)
    loss1 = (dist1.detach() * w).sum() / (w.sum() + 1e-12)
    loss2 = dist2.detach().mean()
    surrogate_loss = loss1 + loss2
    return hard_loss, surrogate_loss


def compute_d2_psnr(
    ref: torch.Tensor,
    rec: torch.Tensor,
    k_normal: int = 16,
    peak_mode: str = "union_bbox_diag",
    final_w: torch.Tensor = None,
    use_torch_ops: bool = False,
):
    """
    ref, rec: [B,3,N]
    final_w:
        - None      : 従来計算
        - [B,1,N_ref] or [B,N_ref]
          ref->rec 側の d2 を重み付き平均する
          chamfer_l2_loss と同じ思想
    """

    assert isinstance(ref, torch.Tensor) and isinstance(rec, torch.Tensor)
    assert ref.ndim == 3 and rec.ndim == 3

    device = ref.device

    ref_xyz = ref.permute(0, 2, 1).contiguous()  # [B,N_ref,3]
    rec_xyz = rec.permute(0, 2, 1).contiguous()  # [B,N_rec,3]

    B, N_ref, _ = ref_xyz.shape
    _, N_rec, _ = rec_xyz.shape

    if N_ref == 0 or N_rec == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    if use_torch_ops or o3d is None or faiss is None:
        ref_normals = estimate_normals_pca(ref, k=k_normal).permute(0, 2, 1).contiguous()
        rec_normals = estimate_normals_pca(rec, k=k_normal).permute(0, 2, 1).contiguous()

        # ref -> rec
        ref_to_rec_idx = _torch_knn_idx(ref_xyz, rec_xyz, 1).squeeze(-1)  # [B,N_ref]
        # rec -> ref
        rec_to_ref_idx = _torch_knn_idx(rec_xyz, ref_xyz, 1).squeeze(-1)  # [B,N_rec]
    else:
        ref_normals = estimate_normals_open3d(ref, k=k_normal).permute(0, 2, 1).contiguous()
        rec_normals = estimate_normals_open3d(rec, k=k_normal).permute(0, 2, 1).contiguous()

        # ref -> rec
        ref_to_rec_idx = _faiss_knn_idx(ref_xyz, rec_xyz, 1).squeeze(-1)  # [B,N_ref]
        # rec -> ref
        rec_to_ref_idx = _faiss_knn_idx(rec_xyz, ref_xyz, 1).squeeze(-1)  # [B,N_rec]

    ref_base = (torch.arange(B, device=device).view(B, 1) * N_ref)
    rec_base = (torch.arange(B, device=device).view(B, 1) * N_rec)

    ref_xyz_flat = ref_xyz.reshape(B * N_ref, 3)
    rec_xyz_flat = rec_xyz.reshape(B * N_rec, 3)
    ref_n_flat   = ref_normals.reshape(B * N_ref, 3)
    rec_n_flat   = rec_normals.reshape(B * N_rec, 3)

    # -------------------------
    # 1. ref -> rec の d2
    # -------------------------
    ref_to_rec_flat = (ref_to_rec_idx + rec_base).reshape(-1)
    nn_rec_xyz = rec_xyz_flat[ref_to_rec_flat].reshape(B, N_ref, 3)
    nn_rec_n   = rec_n_flat[ref_to_rec_flat].reshape(B, N_ref, 3)

    diff_bwd = ref_xyz - nn_rec_xyz
    proj_bwd = (diff_bwd * nn_rec_n).sum(dim=2)
    bwd_sq   = proj_bwd * proj_bwd

    if final_w is not None:
        if final_w.dim() == 3:
            w = final_w.squeeze(1)   # [B,N_ref]
        else:
            w = final_w

        w = w.clamp(0.0, 1.0).to(bwd_sq.dtype)

        if w.shape[1] != N_ref:
            raise ValueError(
                f"final_w shape mismatch: got {tuple(w.shape)}, expected second dim = {N_ref}"
            )

        bwd_mse = (bwd_sq * w).sum() / (w.sum() + 1e-12)
    else:
        bwd_mse = bwd_sq.mean()

    # -------------------------
    # 2. rec -> ref の d2
    # -------------------------
    rec_to_ref_flat = (rec_to_ref_idx + ref_base).reshape(-1)
    nn_ref_xyz = ref_xyz_flat[rec_to_ref_flat].reshape(B, N_rec, 3)
    nn_ref_n   = ref_n_flat[rec_to_ref_flat].reshape(B, N_rec, 3)

    diff_fwd = rec_xyz - nn_ref_xyz
    proj_fwd = (diff_fwd * nn_ref_n).sum(dim=2)
    fwd_sq   = proj_fwd * proj_fwd

    fwd_mse = fwd_sq.mean()

    mse = 0.5 * (bwd_mse + fwd_mse)

    if mse <= 1e-30:
        return torch.tensor(float("inf"), dtype=torch.float32, device=device)

    if peak_mode == "union_bbox_diag":
        all_pts = torch.cat([ref_xyz, rec_xyz], dim=1)
    else:
        all_pts = ref_xyz

    mins = all_pts.amin(dim=1)
    maxs = all_pts.amax(dim=1)
    peak = torch.linalg.norm(maxs - mins, dim=1)
    peak2 = (peak * peak).mean()

    psnr = 10.0 * torch.log10(peak2 / (mse + 1e-8))
    return psnr

# def compute_d2_psnr(ref, rec, k_normal=16, peak_mode="union_bbox_diag"):
#     """
#     ref, rec:
#         - torch.Tensor [B,3,N] または [1,3,N]
#         - numpy (N,3)
#         - str (PLY path)

#     戻り値:
#         torch scalar（deviceは入力Tensorに合わせる）
#     """

#     # -------------------------
#     # 1. Tensorをnumpyへ安全に変換
#     # -------------------------
#     def _to_numpy_points(x):
#         if isinstance(x, torch.Tensor):
#             # [B,3,N] -> (N,3)
#             if x.ndim == 3:
#                 x = x[0]
#             x = x.transpose(0, 1).contiguous()
#             return x.detach().cpu().numpy().astype(np.float64)
#         elif isinstance(x, str):
#             pcd = o3d.io.read_point_cloud(x)
#             return np.asarray(pcd.points, dtype=np.
