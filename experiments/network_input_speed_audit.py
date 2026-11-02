"""CPU-only alternatives for Network training input; no production edits."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ascii_input(path, method='loadtxt', return_color=True):
    with open(path, 'rb') as stream:
        names, count, vertex = [], None, False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Invalid PLY header')
            words = line.decode('ascii').strip().split()
            if words[:1] == ['format'] and words[1] != 'ascii':
                raise ValueError('ASCII prototype only')
            if words[:1] == ['element']:
                vertex = words[1] == 'vertex'
                if vertex:
                    count = int(words[2])
            if vertex and words[:1] == ['property']:
                if len(words) != 3:
                    raise ValueError('Scalar properties only')
                names.append(words[2])
            if words == ['end_header']:
                break
        if method == 'loadtxt':
            values = np.loadtxt(stream, dtype=np.float32, max_rows=count, ndmin=2, comments=None)
        elif method == 'fromstring':
            raw = b''.join(itertools.islice(stream, count))
            values = np.fromstring(raw, dtype=np.float32, sep=' ').reshape(count, len(names))
        else:
            raise ValueError(method)
    xyz = np.stack([values[:, names.index(axis)] for axis in ('x', 'y', 'z')], axis=1).astype(np.float32)
    if not return_color:
        return xyz
    if all(name in names for name in ('red', 'green', 'blue')):
        rgb = np.stack([values[:, names.index(name)] for name in ('red', 'green', 'blue')], axis=1).astype(np.float32)/255.
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    from models.utils.data.dataset import _load_ply_numpy
    paths = [Path(sequence['files'][0]) for sequence in json.loads(options.frames.read_text())['sequences']]
    # Decimal edge cases are checked separately from the integer dataset.
    decimals = ['1.0000000596046447753906251', '1.0000000596046447753906249',
                '-0', '1.1754943508222875e-38', '16777217', '-2.5', '1e-40']
    fixture = options.out/'decimals.ply'
    fixture.write_text('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(decimals)
                       +''.join('%s %s %s\n'%(value,value,value) for value in decimals))
    paths.append(fixture)
    rows = []
    for path in paths:
        expected = _load_ply_numpy(path)
        for repeat in range(options.repeats):
            methods = ['native', 'loadtxt', 'fromstring']
            for method in methods if repeat % 2 == 0 else methods[::-1]:
                started, cpu = time.perf_counter(), time.process_time()
                actual = _load_ply_numpy(path) if method == 'native' else ascii_input(path, method)
                cpu, wall = time.process_time()-cpu, time.perf_counter()-started
                exact = actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
                row = dict(path=str(path), repeat=repeat, method=method, seconds=wall,
                           cpu_seconds=cpu, points=len(actual), exact=exact,
                           different_elements=int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))))
                rows.append(row)
                with (options.out/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
        print(path.name, {method:float(np.median([r['seconds'] for r in rows if r['path']==str(path) and r['method']==method])) for method in methods}, flush=True)


if __name__ == '__main__':
    main()
