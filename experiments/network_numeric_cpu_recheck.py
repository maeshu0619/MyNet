"""CPU-only actual-codec replay against saved GPU results; no codec edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'models/utils/loss'))


def cloud_hash(tensor):
    points = tensor.C[:, 1:].detach().cpu().numpy().astype(np.int64)
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    return hashlib.sha256(points.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--labels', default='gt,edited')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--encode-only', action='store_true')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    # Keep any JIT build outputs private to this experiment.
    os.environ['TORCH_EXTENSIONS_DIR'] = str(options.out/'torch_extensions')
    os.environ.setdefault('MAX_JOBS', '2')
    import torch
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty: CPU only'
    torch.set_num_threads(2)
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))/1024.
    assert available >= 16000, 'Insufficient shared host memory'
    import sparsepcgc_teacher_worker as worker
    codec = ROOT.parent/'compress/octree/SparsePCGC'
    sys.argv = ['probe', '--sparsepcgc-root', str(codec), '--mode', 'dense_lossy',
                '--device', 'cpu', '--scale-m', '8', '--scale-ae', '0', '--scale-sr', '2',
                '--exact-unique-loader', '--exact-morton-stats', '--exact-logical-rate']
    for flag, folder in [('ckptdir', 'dense'), ('ckptdir-sr', 'dense_1stage'), ('ckptdir-ae', 'dense_slne'),
                         ('ckptdir-low', 'sparse_low'), ('ckptdir-high', 'sparse_high'), ('ckptdir-offset', 'sparse_offset')]:
        sys.argv += ['--'+flag, str(codec/'ckpts'/folder/'epoch_last.pth')]
    encoder = worker._build_encoder(worker._parse_args())
    basic = encoder.coder.basic_coder
    original_downscale, side = encoder.coder.downscale, {}
    def capture(*args, **kwargs):
        result = original_downscale(*args, **kwargs)
        side.update(ae=result[1], counts=list(result[2]))
        return result
    encoder.coder.downscale = capture
    reference_rows = [json.loads(line) for line in (options.reference/'timing/records.jsonl').read_text().splitlines()]
    rows = []
    native = sys.modules[type(encoder).__module__]
    labels = options.labels.split(',')
    assert set(labels) <= {'gt', 'edited'}
    for case_index, label in enumerate(labels*options.repeats):
        source = options.reference/('parity/gt.ply' if label == 'gt' else 'timing/edited.ply')
        points, _ = native._read_ply_points(source)
        input_hash = hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest()
        expected = next(row for row in reference_rows if row['input'] == label and row['mode'] == 'native')
        assert input_hash == expected['input_hash']
        folder = options.out/('%s_%d' % (label, case_index))
        result = dict(encoder.encode_one(source, folder))
        stream = next(folder.glob('*.bin'))
        cpu_hash, decoded_count = None, None
        if not options.encode_only:
            low = basic.decode(basic.unpack_bitstream(str(stream)))
            decoded = encoder.coder.upscale(low, side['ae'], list(side['counts']))
            cpu_hash, decoded_count = cloud_hash(decoded), len(decoded)
            del low, decoded
        # Coordinates, rather than stream order, determine geometry.
        row = dict(input=label, case_index=case_index, input_hash=input_hash, cpu_logical_bits=result['file_size'],
                   gpu_logical_bits=expected['logical_bits'], cpu_physical_bits=stream.stat().st_size*8,
                   gpu_physical_bits=expected['physical_bits'], cpu_decoded_hash=cpu_hash,
                   gpu_decoded_hash=expected['decoded_hash'], cpu_decoded_points=decoded_count,
                   gpu_decoded_points=expected['decoded_points'],
                   metrics_equal=all(result[name] == expected['native_metrics'][name]
                                     for name in ('point_count', 'node_count', 'single_child_count')))
        row['bits_equal'] = (row['cpu_logical_bits'] == row['gpu_logical_bits']
                             and row['cpu_physical_bits'] == row['gpu_physical_bits'])
        row['decoded_set_equal'] = cpu_hash == expected['decoded_hash'] if cpu_hash is not None else None
        rows.append(row)
        with (options.out/'results.jsonl').open('a') as output:
            output.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        encoder._cleanup_runtime_cache()
    (options.out/'completed.json').write_text(json.dumps(dict(cases=len(rows))))


if __name__ == '__main__':
    main()
