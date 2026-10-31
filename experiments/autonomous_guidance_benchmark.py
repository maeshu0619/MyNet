"""Verify every real mapped tensor against the original coordinate loops."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import argparse
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    o=p.parse_args();o.out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from models.modules import heuristic_guidance as hg
    from models.utils.pointcloud.autonomous_proposals import _build_autonomous_pool
    torch.set_num_threads(2)
    source=np.load(o.source).astype(np.int64)
    coords=torch.from_numpy(source.T.copy()).unsqueeze(0).cuda()
    args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
        test_route_codec_context=True,sparsepcgc_mode='dense_lossy',test_route_max_operation_ratio=.03,
        sparsepcgc_native_resolution=1023,_autonomous_native_context_path=str(o.out/'context.npz'))
    # Read native observations, write any new core array only to this directory.
    import shutil
    shutil.copy2(o.context,o.out/'context.npz')
    structure=dict(global_voxel_coords=coords)
    exact=_build_autonomous_pool(structure,args,dict(input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),setting_id='audit'))
    like=torch.zeros(1,1,len(source),device='cuda');values={};timings={}
    for name in ('reference','arrays'):
        with mock.patch.object(hg,'_EXACT_GUIDANCE_CACHE',OrderedDict()),\
             mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE',OrderedDict()),\
             mock.patch.object(hg,'_EXACT_GUIDANCE_CPU_CACHE_BYTES',0),\
             mock.patch.object(hg,'_load_exact_guidance_disk',return_value=None),\
             mock.patch.object(hg,'_store_exact_guidance_disk'):
            torch.cuda.synchronize();start=time.perf_counter()
            if name=='reference':
                with mock.patch('models.utils.pointcloud.autonomous_guidance_mapping.map_single_voxel_candidates',return_value=None):
                    value=hg._exact_den6_guidance(structure,args,like,exact)
            else:value=hg._exact_den6_guidance(structure,args,like,exact)
            torch.cuda.synchronize();timings[name]=time.perf_counter()-start
            values[name]=value
    tensor_count=[0]
    def equal(a,b):
        if torch.is_tensor(a):
            assert a.dtype==b.dtype and torch.equal(a,b)
            tensor_count[0]+=1
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:equal(a[key],b[key])
        else:assert a==b
    equal(values['reference'],values['arrays'])
    result=dict(source_points=len(source),pool_counts=exact['proposal_pool_counts'],
        all_tensors_exact=True,tensor_count=tensor_count[0],seconds=timings)
    with (o.out/'comparison.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
