"""Exact CPU statistics for the native teacher, without changing its coder.

Morton ordering turns each octree level into contiguous groups. Counting
distinct children requires no 3-D unique/inverse map or occupancy matrix.
Unsupported coordinate ranges fall back to the native reference function.
"""
import time

import numpy as np


def _spread21(values):
    value = values.astype(np.uint64, copy=False) & np.uint64(0x1fffff)
    for shift, mask in ((32, 0x1f00000000ffff), (16, 0x1f0000ff0000ff),
                        (8, 0x100f00f00f00f00f), (4, 0x10c30c30c30c30c3),
                        (2, 0x1249249249249249)):
        value = (value | (value << np.uint64(shift))) & np.uint64(mask)
    return value


def quantized_morton_stats(points, voxel_size=1., pos_quantscale=1):
    """Return native (unique points, nodes, single children), or None."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    if voxel_size <= 0:
        raise ValueError('voxel_size must be > 0, got {}'.format(voxel_size))
    if not np.isfinite(voxel_size) or pos_quantscale <= 0 or not np.isfinite(points).all():
        return None
    rounded = np.round(points / float(voxel_size))
    if rounded.size and (not np.isfinite(rounded).all()
                         or rounded.min() < -(2.**63) or rounded.max() >= 2.**63):
        return None
    coords = rounded.astype(np.int64)
    # Native code rounds twice, with an int64 conversion in between.
    if pos_quantscale != 1:
        coords = np.round(coords / float(pos_quantscale)).astype(np.int64)
    if not coords.size:
        return 0, 0, 0
    minimum = coords.min(0)
    maximum = coords.max(0)
    span = [int(hi)-int(lo) for lo, hi in zip(minimum, maximum)]
    if any(value < 0 or value >= 2**21 for value in span):
        return None
    coords = coords-minimum
    keys = (_spread21(coords[:, 0]) << np.uint64(2)
            | _spread21(coords[:, 1]) << np.uint64(1)
            | _spread21(coords[:, 2]))
    current = np.unique(keys)
    count = int(current.size)
    depth = max(int(np.ceil(np.log2(max(max(span)+1, 2)))), 1)
    nodes = single = 0
    for _ in range(depth):
        parent = current >> np.uint64(3)
        starts = np.r_[0, np.flatnonzero(parent[1:] != parent[:-1])+1]
        sizes = np.diff(np.r_[starts, len(parent)])
        nodes += int(starts.size)
        single += int(np.count_nonzero(sizes == 1))
        current = parent[starts]
    return count, nodes, single


def make_pointcloud_stats(reader, reference):
    """Installable worker-local wrapper. Native PLY reader/rounding are retained."""
    def pointcloud_stats(path, voxel_size, pos_quantscale):
        started = time.perf_counter()
        points, header_count = reader(path)
        result = quantized_morton_stats(points, voxel_size, pos_quantscale)
        if result is None:
            return reference(path, voxel_size, pos_quantscale)
        count, nodes, single = result
        return dict(header_point_count=int(header_count), point_count=count,
                    node_count=nodes, single_child_count=single,
                    stats_time=time.perf_counter()-started)
    return pointcloud_stats


def install_pointcloud_stats(module):
    """Patch this worker's module object only, keeping a native fallback."""
    reader = getattr(module, '_read_ply_points', None)
    reference = getattr(module, '_pointcloud_stats', None)
    if not callable(reader) or not callable(reference):
        return False
    if getattr(reference, '_mynet_exact_morton_stats', False) is True:
        return True
    wrapper = make_pointcloud_stats(reader, reference)
    wrapper._mynet_exact_morton_stats = True
    module._pointcloud_stats = wrapper
    return True
