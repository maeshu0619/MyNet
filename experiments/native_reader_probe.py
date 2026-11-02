"""CPU-only exact native PLY reader comparison, with untouched codec sources."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    from models.utils.loss.autonomous_native_loader import make_exact_binary_reader
    sys.path.insert(0, str(ROOT.parent/'compress/octree/SparsePCGC'))
    from data_utils.inout import read_coords as native
    fast = make_exact_binary_reader(native)
    # Finite fractional/negative/extreme values, duplicates and order checks.
    fixture = np.array([[-1.9, 2.9, 0.], [3.5, -4.5, 0.], [-1.9, 2.9, 0.],
                        [-2.**31, 2.**31-128, -0.]], dtype='<f4')
    path = options.out/'fractional.ply'
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
              'property float x\nproperty float y\nproperty float z\nend_header\n') % len(fixture)
    path.write_bytes(header.encode()+fixture.tobytes())
    inputs = list(options.input)+[path]
    rows = []
    for path in inputs:
        expected = native(str(path))
        for repeat in range(options.repeats):
            functions = [('native', native), ('direct', fast)]
            for name, function in (functions if repeat % 2 == 0 else functions[::-1]):
                start = time.perf_counter()
                cpu = time.process_time()
                actual = function(str(path))
                cpu = time.process_time()-cpu
                wall = time.perf_counter()-start
                assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
                rows.append(dict(input=str(path), repeat=repeat, mode=name, wall_seconds=wall,
                                 cpu_seconds=cpu, points=len(actual), exact=True))
        print(path.name, {name: float(np.median([r['wall_seconds'] for r in rows
              if r['input']==str(path) and r['mode']==name])) for name in ('native','direct')}, flush=True)
    (options.out/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
