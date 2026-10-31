"""Measure lossless pool restoration and cyclic-GC overhead on real data."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
from types import SimpleNamespace
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--context',required=True,type=Path)
    o=p.parse_args();o.out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from models.utils.pointcloud.autonomous_proposals import _build_autonomous_pool
    torch.set_num_threads(2)
    shutil.copy2(o.context,o.out/'context.npz')
    source=np.load(o.source).astype(np.int64)
    args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
        test_route_codec_context=True,sparsepcgc_mode='dense_lossy',test_route_max_operation_ratio=.03,
        sparsepcgc_native_resolution=1023,_autonomous_native_context_path=str(o.out/'context.npz'))
    value=_build_autonomous_pool(dict(global_voxel_coords=torch.from_numpy(source.T.copy()).unsqueeze(0)),args,
        dict(input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),setting_id='decode_benchmark'))
    encoded=zlib.compress(pickle.dumps(value,protocol=4),1)
    measurements=[]
    for round_id in range(4):
        for mode in ('standard','batch_gc'):
            gc.collect();before=gc.get_stats();enabled=gc.isenabled()
            start=time.perf_counter()
            try:
                if mode=='batch_gc' and enabled:gc.disable()
                decoded=pickle.loads(zlib.decompress(encoded))
                restored=time.perf_counter()
                if mode=='batch_gc' and enabled:gc.collect(0)
            finally:
                if enabled:gc.enable()
            elapsed=time.perf_counter()-start
            assert decoded==value
            after=gc.get_stats()
            measurements.append(dict(round=round_id,mode=mode,restore_seconds=restored-start,
                total_seconds=elapsed,collections=[b['collections']-a['collections'] for a,b in zip(before,after)]))
            del decoded
    result=dict(pool_counts=value['proposal_pool_counts'],compressed_bytes=len(encoded),all_equal=True,measurements=measurements)
    (o.out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
