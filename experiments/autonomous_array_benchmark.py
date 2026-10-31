"""Compare every real candidate against the original loop implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--context',required=True,type=Path)
    o=parser.parse_args();o.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from models.utils.pointcloud.autonomous_proposals import _build_autonomous_pool
    from models.utils.pointcloud.autonomous_context_proposals import _context_proposals_reference
    from models.utils.pointcloud.autonomous_context_arrays import context_proposals_arrays
    torch.set_num_threads(2)
    source=np.load(o.source).astype(np.int64)
    digest=hashlib.sha256(source.tobytes()).hexdigest()
    seed=int(digest[:16],16)&0x7fffffff
    args=SimpleNamespace(test_route=True,compress='SparsePCGC',heuristic_guidance_mode='ana_den6_online',
        test_route_codec_context=False,test_route_max_operation_ratio=.03,sparsepcgc_native_resolution=1023)
    context=dict(global_voxel_coords=torch.from_numpy(source.T.copy()).unsqueeze(0))
    payload=_build_autonomous_pool(context,args,dict(input_sha256=digest,setting_id='independent_benchmark'))
    spatial=payload['operation_edit_units'];budget=max(map(len,spatial.values()))
    results={};measurements={}
    for name,function in (('reference',_context_proposals_reference),('arrays',context_proposals_arrays)):
        directory=o.out/name;directory.mkdir();path=directory/'context.npz';shutil.copy2(o.context,path)
        measurements[name]={}
        for mode in ('cold','warm'):
            start=time.perf_counter();value=function(source,path,budget,seed,spatial)
            elapsed=time.perf_counter()-start
            sha=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            measurements[name][mode]=dict(seconds=elapsed,sha256=sha,counts={op:len(v) for op,v in value[0].items()})
            if name=='reference':results[mode]=value
            else:
                assert results[mode]==value,'Candidate/observation/order changed: '+mode
                assert sha==measurements['reference'][mode]['sha256'],'Serialized candidate values changed'
    with (o.out/'comparison.json').open('x') as stream:
        json.dump(dict(source=str(o.source),context=str(o.context),source_points=len(source),
            budget=budget,all_candidates_exact=True,results=measurements),stream,indent=2)
    print(json.dumps(measurements,indent=2))


if __name__=='__main__':main()
