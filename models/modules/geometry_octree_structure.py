"""Conventional octree analysis, sharing only codec-independent descriptors."""
import torch
from .octree_structure import OctreeStructureAnalysis
from ..utils.pointcloud.geometry_octree import build_geometry_octree_context, codec_key, geometry_step


class GeometryOctreeStructureAnalysis(OctreeStructureAnalysis):
    """Same 40-channel interface for the existing diagnosis/Actor/Amount heads.

    Occupancy masks, sibling counts, multilevel topology and local geometry are
    ordinary octree computations, not SparsePCGC-specific operators. Reuse them
    without copying their large implementation; choose the conventional grid
    and a codec-namespaced cache at this boundary.
    """
    @staticmethod
    def _effective_qs(args):
        return geometry_step(args)

    def _fixed_structure_cache_key(self, cache_key):
        key = super()._fixed_structure_cache_key(cache_key)
        return f'geometry_octree_v2|{codec_key(self.args)}|{key}' if key else ''

    def _prebuilt_octree_context(self, pts_xyz, subtree_tree=None, full_octree_context=None):
        context = super()._prebuilt_octree_context(pts_xyz, subtree_tree, full_octree_context)
        if context is None:
            return None
        if not getattr(self.args, 'geometry_octree_joint_policy', True):
            return context
        source = subtree_tree if isinstance(subtree_tree, dict) else full_octree_context
        coords = self._normalize_global_coords_n3(
            self._tree_tensor(source, 'global_voxel_coords', pts_xyz.device, dtype=torch.long),
            point_count=pts_xyz.shape[-1], device=pts_xyz.device)
        parents, inverse = torch.unique(torch.div(coords, 2, rounding_mode='floor'), dim=0, sorted=True, return_inverse=True)
        slot = (coords[:, 0] & 1) * 4 + (coords[:, 1] & 1) * 2 + (coords[:, 2] & 1)
        occupancy = torch.zeros((parents.shape[0], 8), device=coords.device, dtype=torch.bool)
        occupancy[inverse, slot] = True
        codes = (occupancy.long() * (2 ** torch.arange(8, device=coords.device))).sum(1)
        hist = torch.bincount(codes, minlength=256).to(pts_xyz.dtype)
        probability = hist[codes][inverse] / max(int(parents.shape[0]), 1)
        result = context.clone()
        result[:, 3, :] = probability
        nll = -torch.log2(probability.clamp_min(torch.finfo(pts_xyz.dtype).eps)) / 8.
        child_entropy = torch.log2(occupancy.sum(1)[inverse].to(pts_xyz.dtype)) / 3.
        result[:, 4, :] = torch.maximum(child_entropy, nll.clamp(0., 1.))
        return result

    def forward(self, pts_xyz, final_w=None, coord_scale=None, subtree_tree=None,
                full_octree_context=None, octree_input_mode='auto', cache_key=None):
        if full_octree_context is None and subtree_tree is None:
            full_octree_context = build_geometry_octree_context(pts_xyz, self.args, coord_scale)
        return super().forward(pts_xyz, final_w, coord_scale, subtree_tree,
                               full_octree_context, octree_input_mode, cache_key)
