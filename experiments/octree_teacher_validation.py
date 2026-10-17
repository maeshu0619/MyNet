#!/usr/bin/env python
"""Independent parity and rate-concentration audit for the OA research probe."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from octree_codec_research import OAActual
from geometry_micro_edit_probe import unique, load_ply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    q=unique(np.rint(raw/2.))
    q-=q.min(0)
    oa=OAActual('cpu')
    stats,seq,probs=oa.encode(q,detail=True,roundtrip=True)
    teacher_bits=oa.teacher.encode_bits(torch.from_numpy(q.T.copy()).float())['bit']
    if teacher_bits!=stats['bits']:
        raise AssertionError('Arithmetic bits differ from current OA teacher')
    target=seq[:,-1,0]-1
    ce=-np.log2(np.maximum(probs[np.arange(len(seq)),target],1e-30))
    leaves=ce[seq[:,-1,1]==seq[:,-1,1].max()]
    top=np.sort(leaves)[::-1]
    audit=[]
    for fraction in (.0006,.0015,.003,.01):
        k=round(len(q)*fraction)
        audit.append(dict(voxel_budget_percent=100*fraction,symbol_count=k,
                          top_leaf_ce_bits=float(top[:k].sum()),
                          fraction_total_ce_percent=100*float(top[:k].sum()/ce.sum())))
    oa.teacher.qs=2.
    native=oa.teacher.encode_bits(torch.from_numpy(raw.T.copy()).float())
    result=dict(frame=a.frame,canonical_voxels=len(q),common_grid_actual=stats,
                existing_teacher_common_grid_bits=teacher_bits,teacher_parity=True,
                native_OA_raw_min_first_qs2_bits=native['bit'],
                leaf_ce_fraction_percent=100*float(leaves.sum()/ce.sum()),
                leaf_nodes=len(leaves),max_leaf_ce_bits=float(top[0]),top_cost_concentration=audit,
                caveat='CE concentration is frozen-context diagnostic, not an Actual gain bound')
    (folder/'validation.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(folder/'symbol_ce.npz',ce_bits=ce,levels=seq[:,-1,1])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
