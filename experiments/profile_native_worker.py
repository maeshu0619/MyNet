"""Private native-worker entry point; profile the original codec unchanged."""
import cProfile
import json
import os
from pathlib import Path
import sys
import time
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'models/utils/loss'))
import sparsepcgc_teacher_worker as worker

snapshot = os.environ.get('MYNET_NATIVE_SOURCE_SNAPSHOT')
if snapshot:
    source = Path(snapshot)/'models/utils/loss/autonomous_native_loader.py'
    spec = importlib.util.spec_from_file_location('autonomous_native_loader', str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

original_build = worker._build_encoder


def build(args):
    encoder = original_build(args)
    if os.environ.get('MYNET_NATIVE_PROFILE_ENABLED') == '0':
        return encoder
    directory = Path(os.environ['MYNET_NATIVE_PROFILE_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    original = encoder.encode_one
    number = [0]
    stage_times = {}
    def wrap(obj, name, label):
        function = getattr(obj, name)
        def timed(*a, **kw):
            start = time.perf_counter()
            try:
                return function(*a, **kw)
            finally:
                stage_times[label] = stage_times.get(label, 0.) + time.perf_counter()-start
        setattr(obj, name, timed)
    for obj, name, label in (
        (encoder.coder.basic_coder, 'load_data', 'load_data'),
        (encoder.coder, 'downscale', 'downscale'),
        (encoder.coder.basic_coder, 'encode', 'encode'),
        (encoder.coder.basic_coder, 'pack_bitstream', 'pack'),
        (encoder, '_full_cleanup_runtime_cache', 'full_cleanup')):
        wrap(obj, name, label)
    def encode(*a, **kw):
        number[0] += 1
        stage_times.clear()
        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        try:
            result = original(*a, **kw)
        finally:
            profiler.disable()
            profiler.dump_stats(str(directory/('encode_%03d.pstats' % number[0])))
        row = dict(request=number[0], elapsed=time.perf_counter()-start, stages=dict(stage_times),
                   metrics={k:v for k,v in result.items() if k in
                            ('stats_time','all_enc_time','enc_time','point_count','node_count','file_size','wall_time_seconds')
                            or k.startswith('cuda')})
        with (directory/'encodes.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        return result
    encoder.encode_one = encode
    return encoder


worker._build_encoder = build
if __name__ == '__main__':
    worker.main()
