"""Exact aligned-symbol NLL accounting, not a causal voxel-count experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def keys(coords):
    value=np.ascontiguousarray(coords,dtype=np.int64).reshape(-1,3)
    return value.view(np.dtype([('x','<i8'),('y','<i8'),('z','<i8')])).reshape(-1)


def counterfactual_nll(probability,symbols):
    # This reproduces float32 round-to-even in native torchac, rather than
    # approximating its 16-bit CDF with a floating Bernoulli probability.
    middle=np.rint((np.float32(1)-probability.astype(np.float32))*np.float32(65534)).astype(np.int64)+1
    width=np.where(symbols>0,65536-middle,middle)
    return -np.log2(width/65536.)


def export(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def decompose(a,b):
    _,ia,ib=np.intersect1d(keys(a['coords']),keys(b['coords']),assume_unique=True,return_indices=True)
    ra=np.ones(len(a['nll']),dtype=bool);rb=np.ones(len(b['nll']),dtype=bool)
    ra[ia]=False;rb[ib]=False
    changed=a['occupancy'][ia]!=b['occupancy'][ib]
    counter=counterfactual_nll(a['probability'][ia],b['occupancy'][ib])
    old=counterfactual_nll(a['probability'][ia],a['occupancy'][ia])
    if not np.allclose(old,a['nll'][ia],atol=1e-12,rtol=0):
        raise ValueError('Integer CDF reconstruction differs from observed coder')
    label=counter-old;probability=b['nll'][ib]-counter
    result=dict(removed_candidates=int(ra.sum()),added_candidates=int(rb.sum()),
                common_candidates=len(ia),changed_common_symbols=int(changed.sum()),
                changed_cdf_unchanged_symbols=int(np.count_nonzero((a['cdf_probability'][ia]!=b['cdf_probability'][ib])&~changed)),
                removed_NLL=-float(a['nll'][ra].sum()),added_NLL=float(b['nll'][rb].sum()),
                symbol_change_at_old_context_NLL=float(label.sum()),
                context_change_edited_symbols_NLL=float(probability[changed].sum()),
                context_change_unchanged_symbols_NLL=float(probability[~changed].sum()),
                total_NLL_change=float(b['nll'].sum()-a['nll'].sum()))
    parts=sum(result[k] for k in ['removed_NLL','added_NLL','symbol_change_at_old_context_NLL',
                                  'context_change_edited_symbols_NLL','context_change_unchanged_symbols_NLL'])
    result['decomposition_roundoff']=parts-result['total_NLL_change']
    if abs(result['decomposition_roundoff'])>1e-6:raise ValueError('NLL accounting failed')
    n0,n1=len(a['nll']),len(b['nll']);m0,m1=a['nll'].mean(),b['nll'].mean()
    result.update(symmetric_count_term=float((n1-n0)*(m0+m1)/2),
                  symmetric_mean_cost_term=float((m1-m0)*(n0+n1)/2))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    options=parser.parse_args();native=options.root/'native_v3'
    if not (native/'integrity.json').exists():raise ValueError('Trace has not completed')
    out=options.root/'native_analysis';out.mkdir(exist_ok=False)
    stages=[];pairs=[];summary=[]
    for index in range(4):
        directories=[native/('frame%d_%s'%(index,v)) for v in ['baseline','mine']]
        traces=[json.loads((d/'trace.json').read_text()) for d in directories]
        dictionaries=[{(r['level'],r['stage']):r for r in t['stages']} for t in traces]
        if dictionaries[0].keys()!=dictionaries[1].keys():
            raise ValueError('Different numbers of levels: requires a separately labelled topology comparison')
        for key in dictionaries[0]:
            first,second=[d[key] for d in dictionaries]
            arrays=[np.load(str(d/r['array_file'])) for d,r in zip(directories,[first,second])]
            result=decompose(*arrays)
            pairs.append(dict(frame=index,level=key[0],stage=key[1],**result))
        row=dict(frame=index,name=traces[0]['case']['frame'])
        for variant,trace in zip(['baseline','mine'],traces):
            physical,streams=trace['stream_measurement'];main=streams[0]['bytes']*8
            row.update({variant+'_points':trace['case']['points'],
                        variant+'_candidates':sum(r['candidates'] for r in trace['stages']),
                        variant+'_occupied_symbols':sum(r['occupied'] for r in trace['stages']),
                        variant+'_NLL':trace['integer_cdf_NLL'],
                        variant+'_physical_bits':physical,
                        variant+'_arithmetic_bits':trace['arithmetic_payload_bits'],
                        variant+'_main_container_and_coarsest_bits':main-trace['arithmetic_payload_bits'],
                        variant+'_serialized_side_bits':physical-main})
            for r in trace['stages']:stages.append(dict(frame=index,variant=variant,**r))
        for field in ['removed_NLL','added_NLL','symbol_change_at_old_context_NLL',
                      'context_change_edited_symbols_NLL','context_change_unchanged_symbols_NLL',
                      'removed_candidates','added_candidates','changed_cdf_unchanged_symbols',
                      'symmetric_count_term','symmetric_mean_cost_term']:
            row[field]=sum(r[field] for r in pairs if r['frame']==index)
        row['delta_bits']=row['mine_physical_bits']-row['baseline_physical_bits']
        row['delta_NLL']=row['mine_NLL']-row['baseline_NLL']
        row['physical_rate_change_percent']=100*(row['mine_physical_bits']/row['baseline_physical_bits']-1)
        summary.append(row)
    export(out/'stages.csv',stages);export(out/'aligned_decomposition.csv',pairs)
    export(out/'frames.csv',summary)
    (out/'summary.json').write_text(json.dumps(dict(frames=summary,
        note='Exact accounting under baseline-CDF-first ordering. Count/mean split is descriptive, not causal.'),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
