import torch
import math
import logging
import os
import sys
import importlib.util
import traceback
try:
    from einops import rearrange
except ModuleNotFoundError:
    from models.utils.misc.einops_compat import rearrange
pointops = None
KNN_BACKEND = "chunked_torch_cdist"
POINTOPS_AVAILABLE = False
POINTOPS_IMPORT_ERROR = ""
POINTOPS_IMPORT_TRACEBACK = ""
_ALLOW_SLOW_KNN_FALLBACK = True
_POINTOPS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pointops"))
_POINTOPS_SRC = os.path.join(_POINTOPS_ROOT, "src")
_POINTOPS_LEGACY_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "pointops", "src"))
for _path in (_POINTOPS_SRC, _POINTOPS_LEGACY_SRC):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)
_use_pointops = os.environ.get("MYNET_USE_POINTOPS_CUDA", "auto").strip().lower()
_allow_pointops_build = os.environ.get("MYNET_POINTOPS_ALLOW_BUILD", "0").strip().lower() in {"1", "true", "yes"}
if _use_pointops in {"1", "true", "yes", "auto"}:
    try:
        # pointops.py 側で pointops_cuda が無ければ JIT build が走る
        from models.pointops.functions import pointops as _pointops_module

        pointops = _pointops_module
        KNN_BACKEND = "pointops_cuda"
        POINTOPS_AVAILABLE = True
        POINTOPS_IMPORT_ERROR = ""
        POINTOPS_IMPORT_TRACEBACK = ""
    except Exception as exc:
        pointops = None
        KNN_BACKEND = "chunked_torch_cdist"
        POINTOPS_AVAILABLE = False
        POINTOPS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        POINTOPS_IMPORT_TRACEBACK = traceback.format_exc()
        logging.warning(
            "pointops CUDA extension could not be loaded; falling back to torch cdist. "
            "Traceback:\n%s",
            POINTOPS_IMPORT_TRACEBACK,
        )
else:
    POINTOPS_IMPORT_ERROR = "pointops disabled by MYNET_USE_POINTOPS_CUDA"
    POINTOPS_IMPORT_TRACEBACK = POINTOPS_IMPORT_ERROR
import numpy as np
import random
try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from torch.autograd import grad
try:
    from einops import rearrange, repeat
except ModuleNotFoundError:
    from models.utils.misc.einops_compat import rearrange, repeat
try:
    from sklearn.neighbors import NearestNeighbors
except ModuleNotFoundError:
    NearestNeighbors = None
try:
    from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
    chamfer_dist = chamfer_3DDist()
except Exception as exc:
    chamfer_dist = None
    logging.warning("Chamfer3D extension unavailable in utils_repkpu: %s", exc)


def pointops_diagnostics():
    return {
        "pointops_available": bool(POINTOPS_AVAILABLE),
        "knn_backend": str(KNN_BACKEND),
        "pointops_import_error": str(POINTOPS_IMPORT_ERROR),
        "pointops_import_traceback": str(POINTOPS_IMPORT_TRACEBACK),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def configure_knn_backend(args=None, writer=None):
    global _ALLOW_SLOW_KNN_FALLBACK
    allow_slow = bool(getattr(args, "allow_slow_knn_fallback", True)) if args is not None else True
    _ALLOW_SLOW_KNN_FALLBACK = bool(allow_slow)
    if args is not None:
        setattr(args, "pointops_available", bool(POINTOPS_AVAILABLE))
        setattr(args, "knn_backend", str(KNN_BACKEND))
        setattr(args, "pointops_import_error", str(POINTOPS_IMPORT_ERROR))
    diag = pointops_diagnostics()
    msg = (
        "KNNBackend: "
        f"pointops_available={diag['pointops_available']}, "
        f"knn_backend={diag['knn_backend']}, "
        f"allow_slow_knn_fallback={bool(allow_slow)}, "
        f"torch={diag['torch_version']}, torch_cuda={diag['torch_cuda_version']}, "
        f"cuda_available={diag['torch_cuda_available']}"
    )
    if writer is not None and hasattr(writer, "write"):
        writer.write(msg)
        if not bool(POINTOPS_AVAILABLE):
            writer.write("PointopsImportError: " + str(POINTOPS_IMPORT_ERROR))
            writer.write("PointopsImportTraceback:\n" + str(POINTOPS_IMPORT_TRACEBACK))
    else:
        logging.warning(msg)
    if not bool(POINTOPS_AVAILABLE) and not bool(allow_slow):
        raise RuntimeError(
            "pointops CUDA extension is unavailable and --allow_slow_knn_fallback=False. "
            "Fix/rebuild pointops_cuda instead of silently using chunked_torch_cdist. "
            f"Import error: {POINTOPS_IMPORT_ERROR}"
        )
    return str(KNN_BACKEND)


def _raise_if_slow_knn_fallback(op_name):
    if pointops is None and not bool(_ALLOW_SLOW_KNN_FALLBACK):
        raise RuntimeError(
            f"{op_name} would use slow chunked_torch_cdist fallback while "
            "--allow_slow_knn_fallback=False. pointops_cuda import error: "
            f"{POINTOPS_IMPORT_ERROR}"
        )






def set_seed(seed, deterministic=False):
    """
    Pytorch・Numpy・Pythonの乱数を全て固定し、
    同じコード、重み、データを用いた場合に
    同じ結果が出力されるようにする
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    deterministic = bool(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def index_points(pts, idx, chunk=262144):
    """
    chunk: 1回のgatherで処理するインデックス数の上限
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]

    reshape = False
    if idx.dim() == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)

    out = []
    for start in range(0, idx.shape[1], chunk):
        part = idx[:, start:start+chunk]                 # (B, chunk)
        part_expand = part.unsqueeze(1).expand(-1, fdim, -1)
        out.append(torch.gather(pts, 2, part_expand))    # (B, C, chunk)

    res = torch.cat(out, dim=2)

    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res



def FPS(pts, fps_pts_num):
    """
    FPS（Furthest Point Sampling）を計算
    すでに選ばれた点から最も遠い点を順番に取る
    """
    # input: (b, 3, n)

    if pts.dim() != 3 or pts.shape[1] < 3:
        raise ValueError("FPS expects pts with shape [B, C, N] and C >= 3")
    if pts.shape[-1] <= 0:
        raise ValueError("FPS received an empty point set")

    fps_pts_num = int(min(max(int(fps_pts_num), 1), pts.shape[-1]))

    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    if pointops is not None and pts_trans.is_cuda:
        sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    else:
        _raise_if_slow_knn_fallback("FPS")
        B, N, _ = pts_trans.shape
        sample_idx = torch.zeros((B, fps_pts_num), device=pts.device, dtype=torch.long)
        dist = pts_trans.new_full((B, N), float("inf"))
        farthest = torch.zeros((B,), device=pts.device, dtype=torch.long)
        batch_idx = torch.arange(B, device=pts.device)
        for i in range(fps_pts_num):
            sample_idx[:, i] = farthest
            centroid = pts_trans[batch_idx, farthest, :].unsqueeze(1)
            dist = torch.minimum(dist, torch.sum((pts_trans - centroid) ** 2, dim=-1))
            farthest = torch.max(dist, dim=1).indices
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    if pts.dim() != 3 or center_pts.dim() != 3:
        raise ValueError("get_knn_pts expects pts and center_pts with shape [B, C, N/M]")

    b, c, n = pts.shape
    _, c_center, m = center_pts.shape
    if c < 3 or c_center < 3:
        raise ValueError("get_knn_pts expects xyz tensors with at least 3 channels")
    if n <= 0 or m <= 0:
        raise ValueError(f"get_knn_pts received an empty tensor: n={n}, m={m}")

    k_eff = int(min(max(int(k), 1), n))

    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    if pointops is not None and pts_trans.is_cuda and center_pts_trans.is_cuda:
        with torch.no_grad():
            knn_idx = pointops.knnquery_heap(k_eff, pts_trans, center_pts_trans).long()
    else:
        _raise_if_slow_knn_fallback("get_knn_pts")
        # Avoid materializing [B, M, N] for full clouds.  The chunk size is
        # chosen by an element budget, so memory is bounded even when N is large.
        max_elems = int(os.environ.get("MYNET_KNN_MAX_ELEMS", 16 * 1024 * 1024))
        chunk = max(1, min(m, max_elems // max(n * max(b, 1), 1)))
        idx_chunks = []
        with torch.no_grad():
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                dist = torch.cdist(center_pts_trans[:, start:end, :], pts_trans)
                idx_chunks.append(torch.topk(dist, k=k_eff, largest=False, dim=-1).indices.long())
                del dist
        knn_idx = torch.cat(idx_chunks, dim=1)
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx
    
# def get_knn_pts(k, pts, center_pts, return_idx=False):
#     # (b, c, n) -> (b, n, c)
#     pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()

#     # (b, c, m) -> (b, m, c)
#     center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()

#     knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
#     knn_pts = index_points(pts, knn_idx)  # 形状はindex_points実装に依存

#     if return_idx:
#         return knn_pts, knn_idx
#     else:
#         return knn_pts

# def get_knn_pts(k, pts, center_pts, radius=0.4, return_idx=False):
#     """
#     GPU版KNNの計算
#     角中心の周囲k個の最近傍点を高速に取得
#     # """
#     # # input: (b, 3, n)

#     # # (b, n, 3)
#     # pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
#     # # (b, m, 3)
#     # center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
#     # # (b, m, k)
#     # knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
#     # # (b, 3, m, k)
#     # knn_pts = index_points(pts, knn_idx)

#     # if return_idx == False:
#     #     return knn_pts
#     # else:
#     #     return knn_pts, knn_idx
#     pts_np = rearrange(pts.squeeze(0), 'c n -> n c').cpu().numpy()
#     centers_np = rearrange(center_pts.squeeze(0), 'c m -> m c').cpu().numpy()

#     nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree')
#     nbrs.fit(pts_np)

#     all_indices = []
#     for c in centers_np:
#         idx = nbrs.radius_neighbors([c], return_distance=False)[0]

#         # 半径内に点が多すぎる → k個に制限
#         if len(idx) >= k:
#             idx = idx[:k]
#         # 少なすぎる → 通常KNNで補完
#         else:
#             knn = NearestNeighbors(n_neighbors=k)
#             knn.fit(pts_np)
#             idx = knn.kneighbors([c], return_distance=False)[0]

#         all_indices.append(idx)

#     knn_idx = torch.from_numpy(np.stack(all_indices)).long().cuda()  # (M, k)
#     knn_pts = index_points(pts, knn_idx.unsqueeze(0)).squeeze(0)     # (3, M, k)
#     knn_pts = knn_pts.permute(1, 0, 2).contiguous()                  # (M, 3, k)

#     if return_idx:
#         return knn_pts, knn_idx
#     else:
#         return knn_pts


def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    """
    正規化
    これにより、原点中心・半径1以内に数値を抑えることで、
    ネットワークがスケールに依存しないようにする
    """
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance



        
# def extract_knn_patch(k, pts, center_pts):
#     """
#     KNNの計算
#     これを1つのパッチとみなす
#     """
#     # input : (b, 3, n)

#     # (n, 3)
#     pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
#     pts_np = pts_trans.detach().cpu().numpy()
#     # (m, 3)
#     center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
#     center_pts_np = center_pts_trans.detach().cpu().numpy()
#     knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
#     knn_search.fit(pts_np)
#     # (m, k)
#     knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
#     # (m, k, 3)
#     patches = np.take(pts_np, knn_idx, axis=0)
#     patches = torch.from_numpy(patches).float().cuda()
#     # (m, 3, k)
#     patches = rearrange(patches, 'm k c -> m c k').contiguous()

#     return patches
