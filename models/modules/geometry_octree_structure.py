"""Conventional octree analysis, sharing only codec-independent descriptors."""
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
        return f'geometry_octree_v1|{codec_key(self.args)}|{key}' if key else ''

    def forward(self, pts_xyz, final_w=None, coord_scale=None, subtree_tree=None,
                full_octree_context=None, octree_input_mode='auto', cache_key=None):
        if full_octree_context is None and subtree_tree is None:
            full_octree_context = build_geometry_octree_context(pts_xyz, self.args, coord_scale)
        return super().forward(pts_xyz, final_w, coord_scale, subtree_tree,
                               full_octree_context, octree_input_mode, cache_key)
