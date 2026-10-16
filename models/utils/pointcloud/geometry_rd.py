"""Conventional-codec distortion in voxel units, relative to its no-edit grid.

Raw Chamfer remains the reported quality metric. The constant quantization
distortion must not masquerade as distortion caused by the learned edits.
"""
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from .geometry_octree import is_geometry_octree, quantize_geometry_octree, geometry_step


def enabled(args):
    return is_geometry_octree(args) and bool(getattr(args, 'geometry_octree_joint_policy', True))


def set_geometry_reference(points, args):
    if not enabled(args):
        return
    from scipy.spatial import cKDTree
    xyz = points[0].T.detach().float().cpu().numpy()
    # Build from this exact GT, never a stale training context during validation.
    canonical, meta = quantize_geometry_octree(torch.from_numpy(xyz.T.copy())[None], args, return_metadata=True)
    step = float(meta['effective_qs_tensor'].reshape(-1)[0])
    origin = meta['global_offset_tensor'].numpy().reshape(-1, 3)[0]
    key = (hashlib.blake2b(xyz.tobytes(), digest_size=16).hexdigest(), step, tuple(origin))
    cache = getattr(args, '_geometry_noop_reference_cache', None)
    if cache is None:
        cache = OrderedDict()
        args._geometry_noop_reference_cache = cache
    if key not in cache:
        quantized = np.unique(canonical[0].T.numpy(), axis=0)
        reconstructed = quantized * step + origin
        forward = cKDTree(xyz).query(reconstructed, workers=1)[0]
        backward = cKDTree(reconstructed).query(xyz, workers=1)[0]
        cache[key] = float(np.mean(forward ** 2) + np.mean(backward ** 2))
    cache.move_to_end(key)
    while len(cache) > 1024:
        cache.popitem(last=False)
    args._geometry_noop_chamfer = cache[key]
    args._geometry_rd_voxel_scale = step ** 2


def geometry_penalty(geometry, args):
    if not enabled(args):
        return torch.relu(geometry - float(getattr(args, 'cp_tau_geom', 0.)))
    if not hasattr(args, '_geometry_noop_chamfer'):
        raise RuntimeError('geometry RD reference missing: evaluate geometry before Actual policy')
    reference = float(args._geometry_noop_chamfer)
    scale = max(float(args._geometry_rd_voxel_scale), 1e-12)
    return torch.relu((geometry - reference) / scale - float(getattr(args, 'cp_tau_geom', 0.)))
