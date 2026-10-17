#!/usr/bin/env python
"""Fresh probability/CDF verification for contrasting safe single edits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from context_oracle_core import OAActual,WindowCodec,load_ply,unique,costs


def main():
    p=argparse.ArgumentParser();p.add_argument('--frame',required=True)
    p.add_argument('--benchmark',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    absolute=unique(np.rint(raw/2.));q=absolute-absolute.min(0)
    w=WindowCodec(OAActual('cpu'),q)
    records=[json.loads(x) for x in (Path(a.benchmark)/'results.jsonl').read_text().splitlines()]
    candidates=[r for r in records if r.get('frame')==a.frame and r.get('trial')=='candidate'
                and r['quality_ok'] and r['operation_count']==1 and 'edits' in r]
    cases=[('self_improves_total_worsens',min([r for r in candidates if r['edited_node_cdf_gain']>0 and r['total_cdf_gain']<0],key=lambda r:r['total_cdf_gain'])),
           ('self_and_context_improve',max([r for r in candidates if r['edited_node_cdf_gain']>0 and r['unedited_context_cdf_gain']>0],key=lambda r:r['unedited_context_cdf_gain']))]
    results=[]
    for name,record in cases:
        edits=[tuple(x) for x in record['edits']];row,after=edits[0]
        measured=w.evaluate(edits,actual=True,return_probs=True)
        chunks=measured.pop('_chunks');node_rows=measured.pop('_node_rows')
        if measured['actual_oa']['stream_sha256']!=record['actual_oa']['stream_sha256']:
            raise AssertionError('Fresh representative encode does not reproduce')
        ids,prob,symbols=next(x for x in chunks if row in x[0])
        own=int(np.flatnonzero(ids==row)[0]);new_cost=costs(prob,symbols)
        gains=w.base_cost[ids]-new_cost;unedited=ids!=row
        changed=unedited&(np.abs(gains)>1e-9)
        spatial=np.linalg.norm(w.seq[ids[changed],-1,3:6]-w.seq[row,-1,3:6],axis=1)
        details=[]
        ranked=np.argsort(gains)
        for i in list(ranked[:3])+list(ranked[-3:]):
            if ids[i]==row:
                continue
            target=int(symbols[i])
            details.append(dict(row=int(ids[i]),mask=int(target+1),gain_bits=float(gains[i]),
                raw_probability_before=float(w.probs[ids[i],target]),
                raw_probability_after=float(prob[i,target]),
                cdf_bits_before=float(w.base_cost[ids[i]]),cdf_bits_after=float(new_cost[i])))
        result=dict(case=name,benchmark_id=record['id'],edits=record['edits'],counts=record['counts'],
            mask_before=format(int(w.seq[row,-1,0]),'08b'),mask_after=format(after,'08b'),
            raw_probability_before=float(w.probs[row,w.symbols[row]]),
            raw_probability_after_frozen=float(w.probs[row,after-1]),
            raw_probability_after_recomputed=float(prob[own,after-1]),
            own_cdf_bits_before=float(w.base_cost[row]),own_cdf_bits_after=float(new_cost[own]),
            window_fraction=row%w.bptt/w.bptt,
            changed_context_spatial_distance=dict(min=float(spatial.min()),median=float(np.median(spatial)),max=float(spatial.max())),
            unedited_node_examples=details,quality=record['quality'],gpcc_gain=record['gpcc_gain'],**measured)
        results.append(result);print(json.dumps(result),flush=True)
    (folder/'cases.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
