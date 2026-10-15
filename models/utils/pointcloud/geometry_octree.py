"""Conventional G-PCC/OctAttention grids (no SparsePCGC quantizer or teacher)."""
import json
import os
import torch


def codec_key(args):
    return str(getattr(args, 'compress', '')).lower().replace('-', '').replace('_', '').replace(' ', '')


def is_geometry_octree(args):
    return codec_key(args) in {'gpcc', 'octattention'}


def geometry_step(args):
    if codec_key(args) == 'gpcc':
        return max(float(getattr(args, 'gpcc_effective_qs', 1.0)), 1e-9) if getattr(args, 'gpcc_prequantize', True) else 1.0
    return max(float(getattr(args, 'qs', 2.0)), 1e-9)


def geometry_cache_key(args, key):
    return f'geometry_octree_fp32_v1|{codec_key(args)}|q={geometry_step(args)}|{key}' if key else ''


def geometry_actual_cache_fingerprint(args, key):
    def file_identity(path):
        path = os.path.abspath(os.path.expanduser(str(path)))
        if os.path.isdir(path):
            path = os.path.join(path, 'encoder.cfg')
        try:
            stat = os.stat(path)
            return (path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return (path, 'missing')
    paths = ('gpcc_encoder_path', 'gpcc_cfg_dir') if codec_key(args) == 'gpcc' else ('octattention_ckpt',)
    flags = ('gpcc_prequantize', 'gpcc_disable_attribute_coding', 'gpcc_merge_duplicated_points') if codec_key(args) == 'gpcc' else ('octattention_actualcode', 'bptt', 'octattention_teacher_device')
    return json.dumps(dict(schema='geometry_actual_bits_v1', codec=codec_key(args), key=str(key),
        source=file_identity(str(key).split('|', 1)[0]), qs=geometry_step(args),
        files={p:file_identity(getattr(args,p,'')) for p in paths},
        flags={p:getattr(args,p,None) for p in flags}), sort_keys=True)


def quantize_geometry_octree(points, args, coord_scale=None, offset=None, return_metadata=False):
    if points.ndim != 3 or points.shape[1] != 3 or points.shape[-1] == 0:
        raise ValueError('geometry octree requires nonempty [B,3,N] points')
    xyz = points.float()
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('geometry octree input contains non-finite coordinates')
    scale = torch.as_tensor(1.0 if coord_scale is None else coord_scale, device=xyz.device, dtype=xyz.dtype).reshape(-1, 1, 1)
    step = xyz.new_full((xyz.shape[0], 1, 1), geometry_step(args)) / scale.clamp_min(1e-9)
    if offset is not None:
        origin = torch.as_tensor(offset, device=xyz.device, dtype=xyz.dtype).reshape(-1, 3, 1)
        coords = torch.round((xyz - origin) / step).long()
    elif codec_key(args) == 'gpcc':
        # Exactly GPCCGeometryEncoder: round absolute, then integer bbox shift.
        absolute = torch.round(xyz / step).long()
        minimum = absolute.amin(dim=2, keepdim=True)
        coords = absolute - minimum
        origin = minimum.to(xyz.dtype) * step
    else:
        # Exactly OctAttention dataPrepare: subtract the *raw* minimum first.
        origin = xyz.amin(dim=2, keepdim=True)
        # Its numpy path promotes float32 PLY values to float64 before this
        # subtraction. Match half-bin boundaries rather than rounding early.
        coords = torch.round((xyz.double() - origin.double()) / step.double()).long()
    meta = dict(codec=codec_key(args), quant_mode='geometry_octree_round',
                global_qs=geometry_step(args), effective_qs_tensor=step,
                global_offset=origin, global_offset_tensor=origin,
                voxel_size=geometry_step(args), pos_quantscale=1)
    return (coords, meta) if return_metadata else coords


def build_geometry_octree_context(points, args, coord_scale=None):
    coords, meta = quantize_geometry_octree(points, args, coord_scale, return_metadata=True)
    return dict(octree_context_scope='full_cloud', octree_input_mode='full_cloud',
                geometry_octree_meta=meta, geometry_codec=codec_key(args),
                global_voxel_coords=coords, full_global_voxel_coords=coords,
                full_occupied_voxel_coords=coords, global_qs=meta['effective_qs_tensor'],
                global_offset=meta['global_offset_tensor'])


def restore_geometry_octree_points(coords, meta=None, args=None, center=None,
                                   unique=True, dtype=None, device=None):
    # Use only conventional metadata.  In particular, SparsePCGC voxel_size,
    # posQuantscale and center flags must never change this codec's grid.
    from .sparsepcgc_voxel import normalize_voxel_coords_b3n, unique_voxel_coords_batched
    coords = normalize_voxel_coords_b3n(coords, device=device)
    count_before = int(coords.shape[-1])
    if unique:
        result = unique_voxel_coords_batched(coords)
        coords, mask, counts = result['coords'], result['valid_mask'], result['counts']
    else:
        mask = torch.ones((coords.shape[0], coords.shape[-1]), device=coords.device, dtype=torch.bool)
        counts = mask.sum(dim=1)
    meta = meta or {}
    out_dtype = dtype or torch.float32
    step = torch.as_tensor(meta.get('effective_qs_tensor', meta.get('global_qs', geometry_step(args))), device=coords.device, dtype=out_dtype).reshape(-1, 1, 1)
    origin = torch.as_tensor(meta.get('global_offset_tensor', meta.get('global_offset', [0., 0., 0.])), device=coords.device, dtype=out_dtype).reshape(-1, 3, 1)
    xyz = origin + coords.to(out_dtype) * step
    return xyz, dict(restore_input_points=count_before, restore_output_points=int(coords.shape[-1]),
                     restore_center=False, restore_unique=bool(unique), restore_valid_mask=mask,
                     restore_counts=counts, restore_has_meta=bool(meta))


def align_geometry_context(context, node_coords):
    if not isinstance(context, dict):
        return context
    out = dict(context)
    # qs>1 merges raw points. The structure/Actor rows must follow the unique
    # node order, never a truncated/padded copy of the original raw-point map.
    out['global_voxel_coords'] = node_coords
    out['full_global_voxel_coords'] = node_coords
    out['full_occupied_voxel_coords'] = node_coords
    return out


def build_geometry_point_features(points, coord_scale, max_points, qs, raw_downsample_factor=1.,
                                  voxel_scale=1., growth=1.5, max_iters=8, **unused):
    """Point Transformer budget reduction, not a SparseTensor codec input.

    Nodes are already canonical physical xyz; do not quantize them a second
    time with a different offset/rounding rule.  Only encoder sampling is coarse.
    """
    from .sparse_tensor import _downsample_sparse_tensor_points
    scale = float(torch.as_tensor(1. if coord_scale is None else coord_scale).reshape(-1)[0])
    step = float(qs) * float(voxel_scale) / max(scale, 1e-9)
    count = int(points.shape[-1])
    target = max(int(round(count / max(float(raw_downsample_factor), 1.))), 1)
    if max_points > 0:
        target = min(target, int(max_points))
    xyz, feat, counts, size, inverse = _downsample_sparse_tensor_points(points, target, step, growth, max_iters)
    return dict(sparse_xyz=points, sparse_feat=points.new_ones((1, count)), coords_xyz=xyz,
                feat=feat, counts=counts, full_to_coarse_idx=inverse, raw_points=count,
                pre_downsample_points=count, voxel_size=size, sparse_voxel_size=step)
