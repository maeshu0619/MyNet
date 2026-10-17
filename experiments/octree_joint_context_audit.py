#!/usr/bin/env python
"""Separate edited-symbol gain from collateral OA context/CDF changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from octree_joint_mask_research import OAActual,load_ply,unique,difference,quality


def coder_costs(probs, symbols):
    from numpyAc.numpyAc import pdf_convert_to_cdf_and_normalize,_convert_to_int_and_normalize
    cdf=_convert_to_int_and_normalize(pdf_convert_to_cdf_and_normalize(probs),True)
    unsigned=cdf.astype(np.uint16).astype(np.int64)
    freq=(unsigned[:,1:]-unsigned[:,:-1])%65536
    if not np.all(freq.sum(1)==65536):
        raise AssertionError('CDF total differs')
    return 16.-np.log2(freq[np.arange(len(symbols)),symbols])


def describe(values):
    return dict(min=float(values.min()),median=float(np.median(values)),
                max=float(values.max()),mean=float(values.mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--joint-results',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--budget',type=float,default=.0015)
    a=p.parse_args()
    torch.set_num_threads(2)
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    oa=OAActual('cpu')
    with (folder/'audit.jsonl').open('w',buffering=1) as log:
        for frame in a.frames:
            raw=load_ply(frame,return_color=False).astype(np.float64)
            absolute=unique(np.rint(raw/2.))
            origin=absolute.min(0)
            q=absolute-origin
            stem=Path(frame).stem
            out=np.load(str(Path(a.joint_results)/(stem+'_%.5f_best.npz'%a.budget)))['canonical_voxels']
            before,seq,probs=oa.encode(q,detail=True)
            after,new_seq,new_probs=oa.encode(out,detail=True)
            if seq.shape!=new_seq.shape or not np.array_equal(seq[:,:,1:],new_seq[:,:,1:]) or not np.array_equal(seq[:,:-1,0],new_seq[:,:-1,0]):
                raise AssertionError('Non-leaf structure changed')
            old_symbols=(seq[:,-1,0]-1).astype(np.int16)
            new_symbols=(new_seq[:,-1,0]-1).astype(np.int16)
            edited=old_symbols!=new_symbols
            old_cost=coder_costs(probs,old_symbols)
            frozen_cost=coder_costs(probs,new_symbols)
            new_cost=coder_costs(new_probs,new_symbols)
            own=old_cost-frozen_cost
            context=frozen_cost-new_cost
            frozen_prob=probs[np.arange(len(seq)),new_symbols]
            actual_prob=new_probs[np.arange(len(seq)),new_symbols]
            stripped=difference(q,difference(q,out))
            result=dict(frame=frame,edited_leaf_rows=int(edited.sum()),
                actual_stream_gain_bits=before['bits']-after['bits'],
                own_symbol_frozen_integer_cdf_gain_bits=float(own.sum()),
                context_integer_cdf_gain_bits=float(context.sum()),
                context_gain_on_unedited_rows=float(context[~edited].sum()),
                context_gain_on_edited_rows=float(context[edited].sum()),
                net_integer_cdf_gain_bits=float((old_cost-new_cost).sum()),
                unedited_rows_with_changed_cdf_cost=int((np.abs(context[~edited])>1e-9).sum()),
                edited_before_probabilities=describe(probs[np.flatnonzero(edited),old_symbols[edited]]),
                edited_after_frozen_probabilities=describe(frozen_prob[edited]),
                edited_after_recomputed_probabilities=describe(actual_prob[edited]),
                baseline_quality=quality(raw,q,origin,2.),
                all_removals_without_destinations_quality=quality(raw,stripped,origin,2.),
                joint_final_quality=quality(raw,out,origin,2.))
            if abs(result['net_integer_cdf_gain_bits']-(result['own_symbol_frozen_integer_cdf_gain_bits']+result['context_integer_cdf_gain_bits']))>1e-6:
                raise AssertionError('Rate decomposition mismatch')
            np.savez_compressed(folder/(stem+'_context.npz'),edited=edited,own_cdf_gain=own,
                                context_cdf_gain=context,old_symbols=old_symbols,new_symbols=new_symbols)
            log.write(json.dumps(result)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
