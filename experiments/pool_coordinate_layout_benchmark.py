"""Compare numeric-identical candidate coordinate layouts and GC costs."""
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


def freeze(pools):
    for pool in pools.values():
        for candidate in pool:
            for field in ('remove_coords','add_coords'):
                candidate[field]=tuple(tuple(row) for row in candidate[field])


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--context',required=True,type=Path)
    o=p.parse_args();o.out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from models.utils.pointcloud.autonomous_proposals import _build_autonomous_pool
    shutil.copy2(o.context,o.out/'context.npz')
    source=np.load(o.source).astype(np.int64)
    args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
        test_route_codec_context=True,sparsepcgc_mode='dense_lossy',test_route_max_operation_ratio=.03,
        sparsepcgc_native_resolution=1023,_autonomous_native_context_path=str(o.out/'context.npz'))
    value=_build_autonomous_pool(dict(global_voxel_coords=torch.from_numpy(source.T.copy()).unsqueeze(0)),args,
        dict(input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),setting_id='layout_benchmark'))
    checksum=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    expected=checksum(value)
    encoded={'lists':zlib.compress(pickle.dumps(value,protocol=4),1)}
    freeze(value['operation_edit_units'])
    assert checksum(value)==expected
    encoded['tuples']=zlib.compress(pickle.dumps(value,protocol=4),1)
    del value
    records=[]
    for round_id in range(3):
        for mode in ('lists','tuples'):
            gc.collect();before=gc.get_stats();gc.disable();start=time.perf_counter()
            try:
                value=pickle.loads(zlib.decompress(encoded[mode]))
                gc.collect(1)
            finally:gc.enable()
            restored=time.perf_counter()-start
            start=time.perf_counter();gc.collect(2);full=time.perf_counter()-start
            pool=value['operation_edit_units'];tracked=sum(gc.is_tracked(c) for a in pool.values() for c in a)
            assert checksum(value)==expected
            records.append(dict(round=round_id,mode=mode,restore=restored,full_gc=full,
                tracked_candidates=tracked,compressed_bytes=len(encoded[mode]),
                collections=[b['collections']-a['collections'] for a,b in zip(before,gc.get_stats())]))
            del value
    (o.out/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))


if __name__=='__main__':main()
