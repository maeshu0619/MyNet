import torch
from models.utils.pointcloud.utils_repkpu import get_knn_pts

def three_nn(xyz1: torch.Tensor, xyz2: torch.Tensor):
    return three_nn_fp(xyz1, xyz2)

# utils_pointnet.py の three_nn を書き換える

def three_nn_fp(xyz1, xyz2):
    """
    xyz1: [B, 3, M]  query
    xyz2: [B, 3, N]  source
    return:
        dist: [B, M, 3]
        idx : [B, M, 3]
    """
    knn_pts, idx = get_knn_pts(
        3,
        xyz2,
        xyz1,
        return_idx=True,
    )  # knn_pts: [B, 3, M, 3], idx: [B, M, 3]

    dist = torch.linalg.norm(
        xyz1.unsqueeze(-1) - knn_pts,
        dim=1,
    )  # [B, M, 3]

    return dist, idx

def three_nn_chunked(xyz1, xyz2, chunk=2048):
    return three_nn_fp(xyz1, xyz2)

def three_interpolate(points: torch.Tensor,
                      idx: torch.Tensor,
                      weight: torch.Tensor):
    """
    points : [B, C, N]
    idx    : [B, M, 3]
    weight : [B, M, 3]

    return:
        interpolated_points : [B, C, M]
    """

    # --- 形状取得 ---
    B, C, N = points.shape
    _, M, K = idx.shape   # K = 3

    # --- safety ---
    if idx.dtype != torch.long:
        idx = idx.long()

    # [B, C, N] -> [B, N, C]
    points_trans = points.transpose(1, 2).contiguous()

    # idx: [B, M, 3] -> [B, M*3]
    idx_flat = idx.reshape(B, M * K)

    # points_trans : [B, N, C]
    # idx_flat     : [B, M*3]
    gathered = torch.gather(
        points_trans,
        dim=1,
        index=idx_flat.unsqueeze(-1).expand(-1, -1, C)
    )  # [B, M*3, C]

    # [B, M*3, C] -> [B, M, 3, C]
    gathered = gathered.view(B, M, K, C)

    # weight: [B, M, 3] -> [B, M, 3, 1]
    weight_expand = weight.unsqueeze(-1)

    # weighted sum over K=3
    interpolated = torch.sum(gathered * weight_expand, dim=2)  # [B, M, C]

    # [B, M, C] -> [B, C, M]
    return interpolated.transpose(1, 2).contiguous()