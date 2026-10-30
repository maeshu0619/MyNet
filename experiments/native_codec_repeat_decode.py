"""Read-only native-source experiment: repeat encode, real decode, compare sets.

Uses private inputs/streams and the original native decoder + SR. Decoder
starts from the packed stream, not encoder-side latent coordinates.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'models/utils/loss'))


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.shape).encode()+value.dtype.str.encode()+value.tobytes()).hexdigest()


def canonical(array):
    xyz = np.asarray(array, dtype=np.int64)
    return xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]


def geometry(gt, reconstructed):
    from scipy.spatial import cKDTree
    def query(tree, values):
        try:
            return tree.query(values, k=1, eps=0., workers=2)[0]
        except TypeError:
            return tree.query(values, k=1, eps=0.)[0]
    a = query(cKDTree(reconstructed), gt)
    b = query(cKDTree(gt), reconstructed)
    return dict(mse_gt_rec=float(np.mean(a*a)), mse_rec_gt=float(np.mean(b*b)),
        d1_mse=max(float(np.mean(a*a)), float(np.mean(b*b))),
        chamfer_sum=float(np.mean(a*a)+np.mean(b*b)), maximum_distance=float(max(a.max(), b.max())))


def mixed_edit(points):
    xyz = np.asarray(points, dtype=np.int64)
    if xyz.min() < 0 or xyz.max() > 1023:
        raise ValueError('This independent mixed-edit fixture uses a 10-bit grid')
    def key(x):
        return (x[:, 0]*1024+x[:, 1])*1024+x[:, 2]
    occupied = key(xyz)
    def targets(start, offset):
        ids = np.arange(start, min(start+8192, len(xyz)))
        candidate = xyz[ids]+np.asarray(offset)
        valid = ((candidate >= 0) & (candidate <= 1023)).all(1) & ~np.isin(key(candidate), occupied)
        return ids[valid][:128], candidate[valid][:128]
    move_ids, moved = targets(0, (1, 0, 0))
    _, added = targets(30000, (0, 1, 0))
    added = added[~np.isin(key(added), key(moved))]
    prune = np.arange(20000, 20128)
    keep = np.ones(len(xyz), dtype=bool)
    keep[move_ids] = False
    keep[prune] = False
    return np.concatenate((xyz[keep], moved, added)).astype(np.float32), dict(
        Add=len(added), Prune=len(prune), Adjust=len(moved), kind='deterministic independent mixed-edit fixture')


def write_input(path, xyz):
    data = np.asarray(xyz, dtype='<f4')
    with path.open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
            'property float x\nproperty float y\nproperty float z\nend_header\n' % len(data)).encode())
        data.tofile(stream)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--include-edited', action='store_true')
    options = parser.parse_args()
    options.out.mkdir(exist_ok=False)
    import torch
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))/1024.
    if available < 12288 or torch.cuda.mem_get_info()[0] < 14*1024**3:
        raise RuntimeError('Insufficient shared capacity for this private full-decoder check')
    import sparsepcgc_teacher_worker as worker
    codec = ROOT.parent/'compress/octree/SparsePCGC'
    sys.argv = ['native_repeat', '--sparsepcgc-root', str(codec), '--mode', 'dense_lossy',
        '--device', 'cuda', '--scale-m', '8', '--scale-ae', '0', '--scale-sr', '2']
    for flag, folder in (('ckptdir', 'dense'), ('ckptdir-sr', 'dense_1stage'), ('ckptdir-ae', 'dense_slne'),
                         ('ckptdir-low', 'sparse_low'), ('ckptdir-high', 'sparse_high'), ('ckptdir-offset', 'sparse_offset')):
        sys.argv += ['--'+flag, str(codec/'ckpts'/folder/'epoch_last.pth')]
    args = worker._parse_args()
    encoder = worker._build_encoder(args)
    import encoder_multiple as native
    from autonomous_native_stats import install_pointcloud_stats
    from autonomous_native_loader import make_exact_loader
    native_stats = native._pointcloud_stats
    basic = encoder.coder.basic_coder
    loader_namespace = basic.load_data.__func__.__globals__
    native_loader = loader_namespace['load_sparse_tensor']
    optimized_loader = make_exact_loader(native_loader)
    capture = {}
    original_downscale = encoder.coder.downscale
    def downscale(*positional, **named):
        result = original_downscale(*positional, **named)
        capture['ae_stream'], capture['counts'] = result[1], list(result[2])
        capture['low_coords'] = array_hash(canonical(result[0].C[:, 1:].detach().cpu().numpy()))
        return result
    encoder.coder.downscale = downscale
    points, _ = native._read_ply_points(options.input)
    inputs = [('gt', np.asarray(points, dtype=np.float32), dict(kind='original GT'))]
    if options.include_edited:
        edited, description = mixed_edit(points)
        inputs.append(('edited', edited, description))
    results = []
    for label, xyz, description in inputs:
        input_file = options.out/(label+'.ply')
        write_input(input_file, xyz)
        for condition in ('native_a', 'native_b', 'stats_only', 'optimized'):
            native._pointcloud_stats = native_stats
            loader_namespace['load_sparse_tensor'] = native_loader
            if condition in ('stats_only', 'optimized'):
                assert install_pointcloud_stats(native)
            if condition == 'optimized':
                loader_namespace['load_sparse_tensor'] = optimized_loader
            folder = options.out/(label+'_'+condition)
            folder.mkdir()
            started = time.perf_counter()
            result = dict(encoder.encode_one(input_file, folder))
            elapsed = time.perf_counter()-started
            path = next(folder.glob('*.bin'))
            content = path.read_bytes()
            streams = basic.unpack_bitstream(str(path))
            base = canonical(basic.bit2num(streams[-1]))
            # Genuine file decode, then native lossy SR using the exact
            # original count side information; no encoder latent reuse.
            decoded_low = basic.decode(streams)
            decoded_low_hash = array_hash(canonical(decoded_low.C[:, 1:].detach().cpu().numpy()))
            decoded = encoder.coder.upscale(decoded_low, capture['ae_stream'], list(capture['counts']))
            reconstructed = canonical(decoded.C[:, 1:].detach().cpu().numpy())
            row = dict(input=label, condition=condition, input_hash=array_hash(xyz), description=description,
                encode_seconds=elapsed, physical_bits=len(content)*8, stream_sha256=hashlib.sha256(content).hexdigest(),
                arithmetic_chunks=[hashlib.sha256(v).hexdigest() for v in streams[:-1]],
                base_order_sha256=hashlib.sha256(streams[-1]).hexdigest(), base_set_sha256=array_hash(base),
                encoded_low_sha256=capture['low_coords'], decoded_low_sha256=decoded_low_hash,
                count_side_information=list(capture['counts']), decoded_points=len(reconstructed),
                decoded_sha256=array_hash(reconstructed), geometry=geometry(points, reconstructed),
                native_metrics={k: result.get(k) for k in ('file_size', 'point_count', 'node_count', 'single_child_count')})
            results.append(row)
            with (options.out/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            np.savez_compressed(folder/'decoded.npz', coords=reconstructed)
            del decoded, decoded_low, reconstructed
            encoder._full_cleanup_runtime_cache()
            gc.collect()
    (options.out/'completed.json').write_text(json.dumps(dict(cases=len(results))))


if __name__ == '__main__':
    main()
