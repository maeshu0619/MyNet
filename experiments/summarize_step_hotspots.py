"""Summarize actual Step time including preprocessing and reference cleanup."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def rows(directory):
    return [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]


def timing(directory):
    data=rows(directory)
    stdout=directory.with_name(directory.name+'_stdout.txt').read_text()
    printed=[float(x) for x in re.findall(r'Epi\d+/Epo\d+/Step\d+:([\d.]+)s',stdout)]
    records=list(csv.DictReader(next((directory/'log').rglob('*_memory_diagnostics.csv')).open()))
    start={int(x['global_step']):float(x['unix_time']) for x in records if x['phase']=='step_after_data_load'}
    full=[dict(step=int(x['global_step'])+1, seconds=float(x['unix_time'])-start[int(x['global_step'])])
          for x in records if x['phase']=='step_after_cleanup' and int(x['global_step']) in start]
    steady=[x['seconds'] for x in full if x['step']>=6]
    return dict(steps=len(data), printed=printed, full_step=full,
        steady_printed_mean=statistics.mean(printed[5:]) if len(printed)>5 else None,
        steady_full_mean=statistics.mean(steady) if steady else None,
        steady_full_median=statistics.median(steady) if steady else None,
        peak_cuda_mib=max(x['peak_mib'] for x in data),
        source_integrity=json.loads((directory/'integrity.json').read_text()))


def trajectory(before,after):
    a,b=rows(before),rows(after)
    result=[]
    for x,y in zip(a,b):
        c,g='current_compression_value','current_geometry_value'
        result.append(dict(step=x['step'],plan_equal=x['plan']==y['plan'],
            counts_equal=x['plan']['selected_counts']==y['plan']['selected_counts'],
            pool_sizes_equal=x['plan']['candidate_pool_sizes']==y['plan']['candidate_pool_sizes'],
            actual_before=x['policy'][c],actual_after=y['policy'][c],
            geometry_before=x['policy'][g],geometry_after=y['policy'][g]))
    return dict(steps=len(result),equal_plans=sum(x['plan_equal'] for x in result),
        equal_counts=sum(x['counts_equal'] for x in result),
        equal_pool_sizes=sum(x['pool_sizes_equal'] for x in result),
        equal_actual=sum(x['actual_before']==x['actual_after'] for x in result),
        equal_geometry=sum(x['geometry_before']==x['geometry_after'] for x in result),
        actual_max_delta=max(abs(x['actual_before']-x['actual_after']) for x in result),
        actual_mean_delta=statistics.mean(x['actual_after']-x['actual_before'] for x in result),
        geometry_max_delta=max(abs(x['geometry_before']-x['geometry_after']) for x in result),rows=result)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    names=('before_profile','after_profile','after_profile_v2','after_profile_v3',
           'light_before','light_after','light_after_immutable','false_before','false_after')
    result=dict(runs={name:timing(args.root/name) for name in names},
        final_comparison=trajectory(args.root/'light_before',args.root/'light_after_immutable'),
        unchanged_control=trajectory(args.root/'before_profile',args.root/'light_before'),
        false_comparison=json.loads((args.root/'false_comparison.json').read_text()),
        false_static=json.loads((args.root/'false_static.json').read_text()),
        coordinate_layout=json.loads((args.root/'layout_benchmark/results.json').read_text()))
    before=result['runs']['light_before']['steady_full_mean']
    after=result['runs']['light_after_immutable']['steady_full_mean']
    result['full_step_reduction_percent']=100.*(1.-after/before)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(before_seconds=before,after_seconds=after,
                         reduction_percent=result['full_step_reduction_percent'],
                         final_comparison={k:v for k,v in result['final_comparison'].items() if k!='rows'}),indent=2))


if __name__=='__main__':main()
