"""Actual 2^3 hybrids of one measured anchor/variant pair, diagnostic only."""
import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import (
    append_json,canonical_points,digest,geometry_with_fit,materialize,write_json,
)
from experiments.sparsepcgc_direction.quality import effect


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    opts=parser.parse_args();out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from models.utils.loss.actual_encoder import build_actual_encoder
    from models.utils.loss.compression import CompressionLossMixin
    args=SimpleNamespace(**json.loads((opts.run/'effective_args.json').read_text()))
    args.sparsepcgc_device='cpu';args.sparsepcgc_tmp_dir=str(out/'tmp')
    args.sparsepcgc_actual_result_cache=False;args._den6_online_training_step_active=False
    args._current_exact_teacher_mode='full_cloud'
    anchor,variant=[torch.load(str(p),map_location='cpu') for p in sorted(opts.run.glob('train_*_plan.pt'))[:2]]
    assert anchor['frame']==variant['frame'] and np.array_equal(anchor['source'],variant['source'])
    ops=list(anchor['order']);chosenplans={op:{} for op in ops}
    for op in ops:
        index={c['candidate_id']:i for i,c in enumerate(anchor['pools'][op])}
        for key,plan in [(0,anchor),(1,variant)]:
            chosenplans[op][key]=[index[plan['pools'][op][i]['candidate_id']] for i in plan['selected'][op]]
    encoder=build_actual_encoder(args)
    def encode(coords):
        state={'final_voxel_coords':torch.as_tensor(coords,dtype=torch.long).T.unsqueeze(0),
            'voxel_step':anchor.get('voxel_step'),'voxel_offset':anchor.get('voxel_offset')}
        xyz,debug=CompressionLossMixin._voxel_state_to_codec_xyz(None,args,state,torch.zeros(1,3,1))
        assert debug['voxel_state_codec_xyz_used']
        return float(encoder.encode_bits(xyz[0])['bit'])
    try:
        base=encode(anchor['source']);records=[]
        for mask in itertools.product((0,1),repeat=3):
            chosen={op:chosenplans[op][mask[i]] for i,op in enumerate(ops)}
            r,a=set(),set();conflicts=0
            for op,indices in chosen.items():
                for i in indices:
                    c=anchor['pools'][op][i];cr=set(map(tuple,c.get('remove_coords') or ()));ca=set(map(tuple,c.get('add_coords') or ()))
                    conflicts+=int(bool((cr|ca)&(r|a)));r|=cr;a|=ca
            if conflicts:
                append_json(out/'rejected.jsonl',dict(mask=mask,conflicts=conflicts));continue
            coords=effect(anchor,chosen)
            if not any(mask):assert np.array_equal(canonical_points(coords),canonical_points(anchor['final']))
            if all(mask):assert np.array_equal(canonical_points(coords),canonical_points(variant['final']))
            bit=encode(coords);g=geometry_with_fit(anchor,coords,chosen,args);C=100*(bit/base-1)
            changed=(len(chosen['Add'])+len(chosen['Prune'])+2*len(chosen['Adjust']))/len(anchor['source'])
            J=args.w_com*C+args.cp_lambda_geom*max(g['training_G_reconstructed']-args.cp_tau_geom,0.)+args.heuristic_guidance_online_edit_tiebreak_weight*changed
            row=dict(mask=dict(zip(ops,mask)),bit=bit,baseline_bits=base,C=C,J=J,geometry=g,
                     counts={op:len(idx) for op,idx in chosen.items()},hash=digest(canonical_points(coords)))
            records.append(row);append_json(out/'hybrids.jsonl',row)
            print('[HYBRID]',mask,bit,C,J,flush=True)
        write_json(out/'summary.json',dict(frame=anchor['frame'],rows=len(records),total_possible=8,
            note='Rejected conflicting hybrids are not encoded; no false causal attribution'))
    finally:encoder.close()


if __name__=='__main__':main()
