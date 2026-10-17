#!/usr/bin/env python
"""Context-exact OA window proposals + global operation-budget DP.

Boundary leaf rows are not editable, so separate fixed attention windows have
no changed symbols in one another's contexts. Includes integer-CDF rate cost.
Not a full search optimum: each window has a small top-gain prefix menu.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from octree_codec_research import OAActual,leaf_plan,quality,unique,load_ply,ActualCodec


def integer_cdf_cost(probs,symbols):
    from numpyAc.numpyAc import pdf_convert_to_cdf_and_normalize,_convert_to_int_and_normalize
    cdf=_convert_to_int_and_normalize(pdf_convert_to_cdf_and_normalize(probs),True)
    unsigned=cdf.astype(np.uint16).astype(np.int64)
    freq=(unsigned[:,1:]-unsigned[:,:-1])%65536
    if not np.all(freq.sum(1)==65536):
        raise AssertionError('CDF frequency normalization mismatch')
    hit=freq[np.arange(len(symbols)),symbols]
    return float((16.-np.log2(hit)).sum())


def optimize(q,raw,origin,oa,seq,probs,budget,report):
    started=time.perf_counter()
    teacher=oa.teacher
    bptt=min(1024,len(seq)-1)
    k=round(len(q)*budget)
    candidates=leaf_plan(q,raw,origin,2.,seq,probs,0.,return_candidates=True)
    groups={}
    for item in candidates:
        gain,row,_,_,_=item
        if row % bptt:
            groups.setdefault(row//bptt,{})
            # First is highest original-context gain for this leaf.
            groups[row//bptt].setdefault(row,item)
    ids,padded=teacher._batchify(seq,bptt,len(seq))
    mask=teacher._generate_square_subsequent_mask(bptt)
    base_symbols=(seq[:,-1,0]-1).astype(np.int16)
    menus=[]
    forward_calls=0
    for index,(group,by_row) in enumerate(sorted(groups.items())):
        items=list(by_row.values())
        start=(group+1)*bptt
        target_ids=ids[start+1:start+bptt+1].reshape(-1).numpy()
        valid=target_ids>=0
        target_ids=target_ids[valid]
        baseline_cost=integer_cdf_cost(probs[target_ids],base_symbols[target_ids])
        options=[(0,0.,[])]
        for count in (1,2,4,8,16,32):
            if count>min(k,len(items)):
                continue
            selected=items[:count]
            inp=padded[start:start+bptt].clone()
            targets=base_symbols[target_ids].copy()
            for _,row,op,remove,add in selected:
                before=int(seq[row,-1,0])
                after=before
                parent=seq[row,-1,3:6]
                for point,adding in ((remove,False),(add,True)):
                    if point is None:
                        continue
                    delta=np.asarray(point)-parent
                    bit=1 << int(delta[0]*4+delta[1]*2+delta[2])
                    after=after|bit if adding else after&~bit
                inp[row%bptt,0,-1,0]=after-1
                targets[row-(group*bptt+1)]=after-1
            with torch.inference_mode():
                output=teacher._model(inp.to(teacher._oa_device),mask,[])
                new_prob=torch.softmax(output.reshape(-1,255),1)[torch.from_numpy(valid).to(teacher._oa_device)].cpu().numpy()
            gain=baseline_cost-integer_cdf_cost(new_prob,targets)
            options.append((count,gain,selected))
            forward_calls+=1
        menus.append(options)
        if (index+1)%10==0:
            report(dict(window_progress=index+1,total_windows=len(groups),forward_calls=forward_calls))
    # Multiple-choice knapsack. Zero menu prevents predicted worsening.
    dp=np.full(k+1,-np.inf)
    dp[0]=0.
    histories=[]
    for options in menus:
        updated=np.full(k+1,-np.inf)
        choices=np.full(k+1,-1,dtype=np.int16)
        for option,(count,gain,_) in enumerate(options):
            candidate=dp[:k+1-count]+gain
            target=updated[count:]
            better=candidate>target
            target[better]=candidate[better]
            choices[count:][better]=option
        dp=updated
        histories.append(choices)
    used=int(dp.argmax())
    predicted=float(dp[used])
    amount=used
    selected=[]
    for options,history in zip(menus[::-1],histories[::-1]):
        choice=int(history[amount])
        if choice<0:
            raise AssertionError('Invalid DP backtrack')
        count,_,items=options[choice]
        selected.extend(items)
        amount-=count
    if amount:
        raise AssertionError('Operation budget mismatch')
    occupied=set(map(tuple,q.tolist()))
    counts=dict(Add=0,Prune=0,Adjust=0)
    for _,_,op,remove,add in selected:
        if remove is not None:
            occupied.remove(remove)
        if add is not None:
            occupied.add(add)
        counts[op]+=1
    out=unique(list(occupied))
    return out,dict(selected_count=used,counts=counts,predicted_cdf_gain_bits=predicted,
                   windows=len(groups),model_window_forwards=forward_calls,
                   optimization_sec=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--budget',type=float,default=.0015)
    a=p.parse_args()
    torch.set_num_threads(2)
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    oa=OAActual('cpu')
    gpcc=ActualCodec(2.,2)
    with (folder/'results.jsonl').open('w',buffering=1) as log:
        for frame in a.frames:
            raw=load_ply(frame,return_color=False).astype(np.float64)
            q=unique(np.rint(raw/2.))
            origin=q.min(0)
            q-=origin
            before,seq,probs=oa.encode(q,detail=True,roundtrip=True)
            base_cdf=integer_cdf_cost(probs,(seq[:,-1,0]-1).astype(np.int16))
            base_gpcc=gpcc.encode(q,verify=True)
            out,audit=optimize(q,raw,origin,oa,seq,probs,a.budget,
                               lambda x:print(json.dumps(x),flush=True))
            after,new_seq,new_probs=oa.encode(out,detail=True,roundtrip=True)
            if (new_seq.shape!=seq.shape or not np.array_equal(new_seq[:,:-1],seq[:,:-1])
                    or not np.array_equal(new_seq[:,-1,1:],seq[:,-1,1:])):
                raise AssertionError('Non-leaf structure or leaf locations changed')
            after_cdf=integer_cdf_cost(new_probs,(new_seq[:,-1,0]-1).astype(np.int16))
            measured_gain=base_cdf-after_cdf
            if abs(measured_gain-audit['predicted_cdf_gain_bits'])>.05:
                raise AssertionError('Window independence / exact rate prediction failed')
            edited_gpcc=gpcc.encode(out,verify=True)
            result=dict(frame=frame,budget=a.budget,actual_operation_percent=100*audit['selected_count']/len(q),
                baseline_oa_bits=before['bits'],edited_oa_bits=after['bits'],
                compression_percent=100*(after['bits']/before['bits']-1.),
                baseline_gpcc_bits=base_gpcc['bits'],edited_gpcc_bits=edited_gpcc['bits'],
                gpcc_compression_percent=100*(edited_gpcc['bits']/base_gpcc['bits']-1.),
                baseline_quality=quality(raw,q,origin,2.),edited_quality=quality(raw,out,origin,2.),
                measured_cdf_gain_bits=measured_gain,**audit)
            log.write(json.dumps(result)+'\n')
            np.savez_compressed(folder/(Path(frame).stem+'_window.npz'),canonical_voxels=out,origin=origin,qs=2.)
            print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
