"""Actual CPU encode parity across a simulated admission event.

No RAM is deliberately exhausted. The physical host must have enough memory;
only the admission reader briefly reports pressure to exercise reclamation.
All outputs belong to a new private directory, and only our worker is closed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    from experiments.verify_test_route import parse
    from models.utils.config import args as config
    from models.utils.data.dataset import load_ply
    from models.utils.loss.actual_encoder import _SparsePCGCActualEncoder
    import torch

    if _SparsePCGCActualEncoder._host_available_mb() < 16384:
        raise RuntimeError('Real CPU verification requires at least 16 GiB available RAM')
    args = parse(config, ['--compression', 'SparsePCGC', '--test_route', 'True',
        '--sparsepcgc_device', 'cpu', '--sparsepcgc_host_pressure_recovery', 'true'])
    options.out.mkdir(parents=True)
    args.sparsepcgc_tmp_dir = str(options.out/'workspace')
    args.sparsepcgc_omp_threads = 2
    args.sparsepcgc_actual_result_cache = False
    class Writer:
        def write(self, message):
            with (options.out/'worker_events.txt').open('a') as stream:
                stream.write(str(message)+'\n')
    encoder = _SparsePCGCActualEncoder(args, Writer())
    xyz = load_ply(str(options.input), return_color=False, loader='numpy')
    # The encoder API is one cloud [3,N]; CompressionLossMixin removes the
    # training batch axis before calling it. Never pass [1,3,N] here.
    points = torch.from_numpy(xyz.T.copy())
    assert points.ndim == 2 and points.shape[0] == 3
    points_digest = hashlib.sha256(points.numpy().tobytes()).hexdigest()
    real_available = encoder._host_available_mb
    rows = []
    try:
        started = time.monotonic()
        first = encoder.encode_bits(points)
        rows.append(dict(kind='before', seconds=time.monotonic()-started, stats=first))
        calls = [0]
        def simulated_pressure():
            calls[0] += 1
            return 5854.9 if calls[0] <= 1 else real_available()
        encoder._host_available_mb = simulated_pressure
        started = time.monotonic()
        second = encoder.encode_bits(points)
        rows.append(dict(kind='after_pressure_cleanup', seconds=time.monotonic()-started, stats=second))
        encoder._host_available_mb = real_available
        started = time.monotonic()
        third = encoder.encode_bits(points)
        rows.append(dict(kind='next_request', seconds=time.monotonic()-started, stats=third))
        metrics = ('bit', 'bpp', 'bpn', 'point_count', 'node', 'single', 'unique_coord_count')
        compared = {key: [row['stats'].get(key) for row in rows] for key in metrics}
        for key in ('bit', 'bpp', 'point_count'):
            assert first[key] == second[key] == third[key], (key, compared[key])
        assert int(first['point_count']) == len(xyz), (first['point_count'], len(xyz))
        assert hashlib.sha256(points.numpy().tobytes()).hexdigest() == points_digest
        assert encoder._worker_launch_count == 1
        result = dict(input=str(options.input), input_sha256=hashlib.sha256(options.input.read_bytes()).hexdigest(),
            input_points=len(xyz), encoder_input_shape=list(points.shape), input_tensor_unchanged=True,
            device='cpu', actual_memory_pressure=False,
            admission_reading='one simulated 5854.9MiB observation; no RAM filling',
            worker_launches=encoder._worker_launch_count, worker_requests=encoder._worker_request_count,
            compared=compared, results=rows, bit_and_point_count_equal=True)
        with (options.out/'results.json').open('x') as stream:
            json.dump(result, stream, indent=2, default=str)
        print('PASS:', compared, flush=True)
    finally:
        encoder._host_available_mb = real_available
        encoder.close()


if __name__ == '__main__':
    main()
