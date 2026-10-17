#!/usr/bin/env python
"""G-PCC raw-grid vs qs=2 baseline; a distortion-changing control, not success."""
import argparse
import json
from pathlib import Path
import numpy as np
from context_oracle_core import ActualCodec,load_ply,unique,quality


def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    (folder/'arguments.json').write_text(json.dumps(vars(a),indent=2))
    codec=ActualCodec(1.,2);results=[]
    for frame in a.frames:
        raw=load_ply(frame,return_color=False).astype(np.float64)
        original=unique(raw);a2=unique(np.rint(raw/2.));origin=a2.min(0);q=a2-origin
        raw_stats=codec.encode(original,verify=True);quant_stats=codec.encode(q,verify=True)
        result=dict(frame=frame,raw_unique_voxels=len(original),qs2_unique_voxels=len(q),
            raw_gpcc=raw_stats,qs2_gpcc=quant_stats,
            qs2_rate_change_from_raw=100*(quant_stats['bits']/raw_stats['bits']-1.),
            raw_quality=quality(raw,original,np.zeros(3),1.),qs2_quality=quality(raw,q,origin,2.),
            geometry_preserving_success=False,
            interpretation='Loss improvement is relative to the already qs=2-quantized baseline, not raw-grid bytes')
        results.append(result);print(json.dumps(result),flush=True)
    (folder/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
