"""Read-only Network inputs with a private, full-content-verified NPY cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    from models.utils.data.dataset import _load_ply_numpy
    paths = [Path(sequence['files'][0]) for sequence in json.loads(options.frames.read_text())['sequences']]
    rows = []
    for number, path in enumerate(paths):
        # Original parser supplies the cache: no new float parser is assumed.
        expected = _load_ply_numpy(path)
        source_digest = digest_file(path)
        data_digest = hashlib.sha256(expected.tobytes()).hexdigest()
        cache_path = options.out/('frame_%d.npy' % number)
        np.save(cache_path, expected, allow_pickle=False)
        for repeat in range(options.repeats):
            started = time.perf_counter()
            assert digest_file(path) == source_digest  # complete source, not just stat/path
            points = np.load(cache_path, allow_pickle=False)
            assert hashlib.sha256(points.tobytes()).hexdigest() == data_digest
            elapsed = time.perf_counter()-started
            assert points.dtype == expected.dtype and points.shape == expected.shape
            assert points.tobytes() == expected.tobytes()
            rows.append(dict(path=str(path), repeat=repeat, seconds=elapsed,
                             source_sha256=source_digest, data_sha256=data_digest,
                             bytes=points.nbytes, exact=True))
        print(path.name, float(np.median([r['seconds'] for r in rows if r['path']==str(path)])), flush=True)
    (options.out/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
