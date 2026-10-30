"""Compare native statistics against exact Morton grouping on real PLY inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    source = ROOT.parent/'compress/octree/SparsePCGC/encoder_multiple.py'
    spec = importlib.util.spec_from_file_location('_native_stats_reference', str(source))
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    from models.utils.loss.autonomous_native_stats import make_pointcloud_stats, quantized_morton_stats
    fast = make_pointcloud_stats(native._read_ply_points, native._pointcloud_stats)
    rows = []
    for path in args.input:
        record = dict(input=str(path))
        for name, function in (('native', native._pointcloud_stats), ('morton', fast)):
            started = time.perf_counter()
            value = function(path, 1., 1)
            record[name] = dict(seconds=time.perf_counter()-started, result=value)
        record['equal'] = all(record['native']['result'][k] == record['morton']['result'][k]
                              for k in ('header_point_count', 'point_count', 'node_count', 'single_child_count'))
        points, _ = native._read_ply_points(path)
        reference_times, fast_times = [], []
        for order in ((False, True), (True, False)):
            for use_fast in order:
                started = time.perf_counter()
                if use_fast:
                    result = quantized_morton_stats(points, 1., 1)
                    fast_times.append(time.perf_counter()-started)
                else:
                    coords = native._quantized_unique_points(points, 1., 1)
                    nodes, single = native._octree_occupancy_stats(coords)
                    result = (len(coords), nodes, single)
                    reference_times.append(time.perf_counter()-started)
                assert result == tuple(record['native']['result'][k] for k in (
                    'point_count', 'node_count', 'single_child_count'))
        record['same_preloaded_points'] = dict(native_seconds=reference_times, morton_seconds=fast_times)
        rows.append(record)
        print(json.dumps(record), flush=True)
        assert record['equal']
    (args.out/'results.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
