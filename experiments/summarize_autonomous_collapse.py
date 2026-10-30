"""Aggregate immutable real-training probes; missing runs are not successes."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize(directory, include_updates):
    from experiments.summarize_autonomous_speed import update_metrics
    rows=[json.loads(x) for x in (directory/'steps.jsonl').read_text().splitlines()]
    stdout=directory.parent/(directory.name+'_stdout.txt')
    text=stdout.read_text() if stdout.is_file() else ''
    wall=[float(x) for x in re.findall(r'Epi\d+/Epo\d+/Step\d+:([\d.]+)s',text)]
    def mean(values):
        return sum(values)/len(values) if values else None
    validation=directory/'final_validation.json'
    detailed=directory/'final_validation_measurements.json'
    integrity=directory/'integrity.json'
    result=dict(steps=len(rows),validation=json.loads(validation.read_text()) if validation.exists() else None,
        validation_measurements=json.loads(detailed.read_text()) if detailed.exists() else None,
        integrity=json.loads(integrity.read_text()) if integrity.exists() else None,
        training_mean_C=mean([r['policy']['current_compression_value'] for r in rows]),
        training_min_C=min(r['policy']['current_compression_value'] for r in rows),
        negative_steps=sum(r['policy']['current_compression_value']<0 for r in rows),
        noop_steps=sum(not any(r['plan']['selected_counts'].values()) for r in rows),
        replay_uses=sum(bool(r['policy'].get('actual_plan_replay',{}).get('used')) for r in rows),
        wall_mean=mean(wall),wall_last32_mean=mean(wall[-32:]),
        wall_steps33_64_mean=mean(wall[32:64]),
        mean_timer_last32={k:mean([r['times'].get(k,0.) for r in rows[-32:]]) for k in rows[-1]['times']},
        last_pool_cache=rows[-1].get('pool_ram_cache'))
    if include_updates:
        updates=update_metrics(directory)
        result['all_gradients_finite']=all(r['finite_gradients'] for r in updates)
        result['selected_gradient_updates']=[r for r in updates if r['step'] in (1,2,8,32,64,128,len(rows))]
        result['gradient_mean_by_role']={role:{key:mean([r['modules'][role][key] for r in updates
            if r['modules'][role][key] is not None]) for key in updates[0]['modules'][role]}
            for role in updates[0]['modules']}
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',required=True,help='Comma-separated completed run directories')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={name:summarize(args.root/name,args.updates) for name in args.runs.split(',')}
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    for name,row in result.items():
        print(name,json.dumps({k:row[k] for k in ('steps','validation','negative_steps','noop_steps','replay_uses','wall_last32_mean')}))


if __name__=='__main__':main()
