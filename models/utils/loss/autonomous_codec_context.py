"""Read-only observations from the native dense coder's first coded level.

Probabilities describe the unchanged GT context, never a counterfactual
Actual improvement. Native coder methods and arithmetic streams are retained.
"""
import json
import os
from pathlib import Path
import tempfile
import hashlib
import time

import numpy as np


def codec_context_enabled(args):
    from ..training.sparsepcgc_autonomous_policy import autonomous_search_enabled
    return (autonomous_search_enabled(args) and bool(getattr(args, 'test_route_codec_context', False))
            and str(getattr(args, 'sparsepcgc_mode', '')).lower() == 'dense_lossy'
            and int(getattr(args, 'sparsepcgc_scale_ae', 0)) == 0
            and float(getattr(args, 'sparsepcgc_voxel_size', 1.)) == 1.
            and int(getattr(args, 'sparsepcgc_pos_quantscale', 1)) == 1)


def prepare_codec_context(args, loss, gt_xyz, cache_key):
    """Use the one initial GT encode also as a frozen native observation.

    No edited alternative is encoded here. Existing GT caches/logs are not
    overwritten: the new native accounting/context fingerprint has its own key.
    """
    args._autonomous_native_context_path = ''
    if not codec_context_enabled(args):
        return None
    import torch
    if gt_xyz.ndim != 3 or gt_xyz.shape[:2] != (1, 3):
        raise ValueError('Native autonomous context requires one full GT cloud')
    if not bool(torch.isfinite(gt_xyz).all()) or not torch.equal(gt_xyz, gt_xyz.round()):
        raise ValueError('Native autonomous context currently requires exact integer GT coordinates')
    sr = int(getattr(args, 'sparsepcgc_scale_sr', 2))
    def values(value):
        return [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(v) for v in str(value).split(',')]
    if (values(getattr(args, 'sparsepcgc_dense_scale_ae_list', '0')) != [0]
            or values(getattr(args, 'sparsepcgc_dense_scale_sr_list', str(sr))) != [sr]):
        raise ValueError('Native autonomous context requires one matching dense RD setting')
    encoder = loss._get_actual_encoder(args)
    fingerprint = encoder._codec_cache_fingerprint()
    from ..pointcloud.autonomous_context_proposals import canonical_voxel_sha
    rows = gt_xyz[0].T.detach().to(device='cpu', dtype=torch.int64).numpy()
    voxel_sha = canonical_voxel_sha(rows)
    metadata = dict(schema='native_cdf_context_v1', input_voxel_sha256=voxel_sha,
                    scale=2**sr, codec_fingerprint=fingerprint)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(str(getattr(args, 'heuristic_guidance_online_cache_dir')))/'native_context_v1'/(key+'.npz')
    cached_gt = loss._get_cached_actual_gt(cache_key)
    started = time.perf_counter()
    encoded = False
    if not path.is_file() or cached_gt is None:
        previous = {k: getattr(args, k, None) for k in (
            '_native_codec_context_output', '_native_codec_context_metadata', '_sparsepcgc_force_fresh_actual_encode')}
        args._native_codec_context_output = str(path) if not path.is_file() else ''
        args._native_codec_context_metadata = metadata
        if not path.is_file():
            args._sparsepcgc_force_fresh_actual_encode = True
        try:
            stats = loss._encode_actual_batch(args, gt_xyz)
        finally:
            for k, value in previous.items():
                if value is None:
                    if hasattr(args, k):delattr(args, k)
                else:setattr(args, k, value)
        if not path.is_file():
            raise RuntimeError('GT encode did not produce its native occupancy context')
        loss._store_cached_actual_gt(cache_key, stats)
        encoded = True
    args._autonomous_native_context_path = str(path)
    args._autonomous_native_context_debug = dict(gt_encoded=encoded, context_file=str(path),
        prepare_seconds=time.perf_counter()-started, scale=2**sr)
    return str(path)


class NativeOccupancyRecorder:
    def __init__(self, encoder):
        self.path = None
        self.metadata = {}
        self.parts = []
        self.level = 0
        basic = encoder.coder.basic_coder
        encode = basic.encode
        forward = basic.model.upsampler.forward

        def observed_encode(*args, **kwargs):
            self.level = 0
            return encode(*args, **kwargs)

        def observed_forward(x_low, x_high=None, *args, **kwargs):
            result = forward(x_low, x_high, *args, **kwargs)
            if self.path is not None and self.level == 0:
                from data_utils.sparse_tensor import sort_sparse_tensor, isin
                from decoder import BinaryArithmeticCoding
                from torchac.torchac import _convert_to_int_and_normalize
                import torch
                bac = BinaryArithmeticCoding()
                for output in result['out_cls_list']:
                    output = sort_sparse_tensor(output)
                    prob = torch.sigmoid(output.F).detach().cpu()
                    cdf = _convert_to_int_and_normalize(bac._get_cdf(prob), True).to(torch.int64).numpy() % 65536
                    cdf = cdf.reshape(-1, 3)
                    cdf[:, -1] = 65536
                    p = np.diff(cdf, axis=1)[:, 1].astype(np.float64)/65536.
                    occupied = isin(output.C, x_high.C).detach().cpu().numpy().reshape(-1).astype(np.uint8)
                    self.parts.append((output.C[:, 1:].detach().cpu().numpy().copy(), p, occupied))
            self.level += 1
            return result

        basic.encode = observed_encode
        basic.model.upsampler.forward = observed_forward

    def begin(self, path, metadata):
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.parts = []

    def finish(self):
        if self.path is None:
            return None
        path = self.path
        try:
            if not self.parts:
                raise RuntimeError('Native occupancy recorder saw no coded-level predictions')
            coords = np.concatenate([x[0] for x in self.parts])
            p = np.concatenate([x[1] for x in self.parts])
            occupied = np.concatenate([x[2] for x in self.parts])
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.npz', delete=False) as f:
                temporary = Path(f.name)
                np.savez_compressed(f, coords=coords, probability=p, occupied=occupied,
                    metadata=json.dumps(self.metadata, sort_keys=True))
            try:
                # An existing observation is immutable. Two workers may race
                # to create the same content-keyed GT cache file.
                os.link(str(temporary), str(path))
            except FileExistsError:
                pass
            finally:
                temporary.unlink()
            return str(path)
        finally:
            self.path = None
            self.parts = []

    def cancel(self):
        self.path = None
        self.parts = []
