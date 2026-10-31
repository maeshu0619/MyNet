"""Independent parity/timing experiment for the native codec's fixed SR pooling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'models/utils/loss'))


def install(encoder, native_module):
    from autonomous_native_loader import lexicographic_unique_rows
    original = encoder._encode_dense_lossy_only
    coder = encoder.coder
    def encode(input_file, bin_path, *, scale_ae, scale_sr):
        if scale_ae != 0 or scale_sr != 2 or encoder.args.voxel_size != 1. or encoder.args.pos_quantscale != 1:
            return original(input_file, bin_path, scale_ae=scale_ae, scale_sr=scale_sr)
        old_load, old_down = coder.basic_coder.load_data, coder.downscale
        token = SimpleNamespace()
        prepared = {}
        def load(path, voxel, position):
            points, count = native_module._read_ply_points(Path(path))
            if (points.shape != (count, 3) or not np.isfinite(points).all() or not len(points)
                    or points.min() < 0 or points.max() >= 2**21 or not (points == np.trunc(points)).all()):
                return old_load(path, voxel, position)
            rows = lexicographic_unique_rows(points.astype(np.int64))
            counts = [len(rows)]
            rows = lexicographic_unique_rows(rows//2)
            counts.append(len(rows))
            rows = lexicographic_unique_rows(rows//2)
            token.shape = (counts[0], 1)
            prepared.update(rows=rows, counts=counts)
            coder.basic_coder.ref_point = None
            return token
        def down(x, scale_AE=1, scale_SR=2):
            if x is not token:
                return old_down(x, scale_AE=scale_AE, scale_SR=scale_SR)
            torch, me = encoder.torch, encoder.ME
            rows = prepared['rows']
            coordinates = torch.zeros((len(rows), 4), dtype=torch.int32)
            coordinates[:, 1:] = torch.from_numpy(rows.astype(np.int32))
            tensor = me.SparseTensor(features=torch.ones((len(rows), 1)).float(),
                                     coordinates=coordinates, tensor_stride=1, device=encoder.device)
            return tensor, None, list(prepared['counts'])
        coder.basic_coder.load_data, coder.downscale = load, down
        try:
            return original(input_file, bin_path, scale_ae=scale_ae, scale_sr=scale_sr)
        finally:
            coder.basic_coder.load_data, coder.downscale = old_load, old_down
    encoder._encode_dense_lossy_only = encode
    return original


def canon(x):
    c = x.C.detach().cpu().numpy()
    f = x.F.detach().cpu().numpy()
    order = np.lexsort(tuple(c[:, i] for i in reversed(range(c.shape[1]))))
    return dict(coords=c[order], features=f[order], stride=list(x.tensor_stride))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--edited', required=True, type=Path)
    o = p.parse_args()
    o.out.mkdir(parents=True, exist_ok=False)
    import torch
    import sparsepcgc_teacher_worker as worker
    from experiments.native_codec_repeat_decode import write_input
    codec = ROOT.parent/'compress/octree/SparsePCGC'
    sys.argv = ['probe', '--sparsepcgc-root', str(codec), '--mode', 'dense_lossy', '--device', 'cuda',
                '--scale-m', '8', '--scale-ae', '0', '--scale-sr', '2',
                '--exact-unique-loader', '--exact-morton-stats', '--exact-logical-rate']
    for flag, folder in (('ckptdir','dense'), ('ckptdir-sr','dense_1stage'), ('ckptdir-ae','dense_slne'),
                         ('ckptdir-low','sparse_low'), ('ckptdir-high','sparse_high'), ('ckptdir-offset','sparse_offset')):
        sys.argv += ['--'+flag, str(codec/'ckpts'/folder/'epoch_last.pth')]
    encoder = worker._build_encoder(worker._parse_args())
    native = sys.modules[type(encoder).__module__]
    original = encoder._encode_dense_lossy_only
    base_encode = encoder.coder.basic_coder.encode
    capture = {}
    def observed(x, *a, **kw):
        capture.update(canon(x))
        return base_encode(x, *a, **kw)
    encoder.coder.basic_coder.encode = observed
    cases = [('source', np.load(o.source)),
             ('edited', torch.load(str(o.edited), map_location='cpu')['final'])]
    records = []
    for name, points in cases:
        file = o.out/(name+'.ply')
        write_input(file, points)
        expected = None
        for round_id, mode in enumerate(('native', 'fused', 'native', 'fused')):
            encoder._encode_dense_lossy_only = original
            if mode == 'fused':
                install(encoder, native)
            folder = o.out/('%s_%d_%s'%(name, round_id, mode))
            start = time.perf_counter()
            result = dict(encoder.encode_one(file, folder))
            elapsed = time.perf_counter()-start
            current = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in capture.items()}
            if expected is None:
                expected = current
            equal = (current['stride']==expected['stride']
                     and np.array_equal(current['coords'], expected['coords'])
                     and np.array_equal(current['features'], expected['features']))
            stream = next(folder.glob('*.bin')).read_bytes()
            record = dict(case=name, mode=mode, round=round_id, elapsed=elapsed,
                          low_tensor_exact=equal, low_points=len(current['coords']),
                          logical_bits=result['file_size'], physical_bits=8*len(stream),
                          stream_sha256=hashlib.sha256(stream).hexdigest(),
                          metrics={k:v for k,v in result.items() if k.startswith('cuda') or k in
                                   ('all_enc_time','enc_time','point_count','node_count','single_child_count')})
            records.append(record)
            with (o.out/'records.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            assert equal
            encoder._full_cleanup_runtime_cache()
    (o.out/'summary.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
