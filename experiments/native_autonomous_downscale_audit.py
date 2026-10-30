"""Trace real dense-lossy input/downscale and rate for saved plan fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'models/utils/loss'))


def canon(x):
    a=np.asarray(x,dtype=np.int64)
    return a[np.lexsort((a[:,2],a[:,1],a[:,0]))]


def sha(x):
    return hashlib.sha256(canon(x).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--research',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--record-context',action='store_true')
    p.add_argument('--source-only',action='store_true')
    o=p.parse_args();o.out.mkdir(exist_ok=False,parents=True)
    import torch
    import sparsepcgc_teacher_worker as worker
    from experiments.sparsepcgc_direction.common import materialize
    from experiments.native_codec_repeat_decode import write_input
    source=np.load(o.research/'source.npy')
    best=torch.load(o.research/'context_oracle/best_011.pth',map_location='cpu')['final']
    d=json.loads(o.reference.read_text());pools=d['ranked_candidate_pools']
    selected=set(d['initial_heuristic_plan']['candidate_ids'])
    orders={op:[i for i,c in enumerate(pool) if c['candidate_id'] in selected] for op,pool in pools.items()}
    counts={op:len(idx) for op,idx in orders.items()}
    reference,chosen=materialize(source,pools,['Add','Prune','Adjust'],counts,orders)
    codec=ROOT.parent/'compress/octree/SparsePCGC'
    sys.argv=['native_audit','--sparsepcgc-root',str(codec),'--mode','dense_lossy',
        '--device','cuda','--scale-m','8','--scale-ae','0','--scale-sr','2']
    if o.record_context:
        sys.argv+=['--exact-logical-rate']
    for flag,folder in (('ckptdir','dense'),('ckptdir-sr','dense_1stage'),('ckptdir-ae','dense_slne'),
        ('ckptdir-low','sparse_low'),('ckptdir-high','sparse_high'),('ckptdir-offset','sparse_offset')):
        sys.argv+=['--'+flag,str(codec/'ckpts'/folder/'epoch_last.pth')]
    encoder=worker._build_encoder(worker._parse_args());capture={}
    downscale=encoder.coder.downscale
    def observed(x,*a,**kw):
        result=downscale(x,*a,**kw)
        raw=x.C[:,1:].cpu().numpy();low=result[0].C[:,1:].cpu().numpy()
        expected=np.unique(raw//4,axis=0)
        capture.update(raw_points=len(raw),raw_sha=sha(raw),low_points=len(low),low_sha=sha(low),
            expected_low_points=len(expected),expected_low_sha=sha(expected),kwargs=kw,counts=list(result[2]))
        return result
    encoder.coder.downscale=observed
    recorder=None
    if o.record_context:
        from autonomous_codec_context import NativeOccupancyRecorder
        recorder=NativeOccupancyRecorder(encoder)
    cases=[('source',source),('context_best',best),('den6_reference',reference)]
    for name,coords in (cases[:1] if o.source_only else cases):
        folder=o.out/name;folder.mkdir();file=folder/'input.ply';write_input(file,coords)
        if recorder is not None:
            recorder.begin(folder/'context.npz',dict(schema='native_cdf_context_v1',scale=4,input_sha=sha(coords)))
        result=dict(encoder.encode_one(file,folder));binfile=next(folder.glob('*.bin'))
        if recorder is not None:recorder.finish()
        record=dict(name=name,input_sha=sha(coords),points=len(coords),capture=dict(capture),
            physical_bits=binfile.stat().st_size*8,
            metrics={k:result.get(k) for k in ('file_size','main_bin_bits','logical_file_size','side_information_bits','enc_time')})
        if name=='den6_reference':record.update(counts={k:len(v) for k,v in chosen.items()})
        with (o.out/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        np.savez_compressed(folder/'coords.npz',coords=coords)
        encoder._full_cleanup_runtime_cache()


if __name__=='__main__':main()
