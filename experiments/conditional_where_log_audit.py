"""Past-only baseline ablation of measured logs, NOT a new training run."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-snapshot', action='store_true', help='Read frozen audit JSONL rather than a live text log')
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for line in options.log.open():
        if line.startswith('AutonomousLearningAudit: '):
            rows.append(json.loads(line.split(': ',1)[1]))
        elif options.audit_snapshot and line.strip():
            rows.append(json.loads(line))
    snapshot = ''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows)
    (options.out/'audit_snapshot.jsonl').write_text(snapshot)
    memory, conditions, observations = {}, {}, []
    for row in rows:
        family = row['credit_family']
        key = (row['frame'],row['context']['context_file'],family)
        c,g = row['policy']['current_compression_value'],row['policy']['current_geometry_value']
        cost = 3*c+150*g  # verified active weights in this input log
        old = memory.get(key,0.)
        values = list(row['structured_where'].values())
        if family == 'where_sampled' and values:
            assert len(set(values)) == 1
            bucket = int(values[0]>=0.)
            token = (key,bucket)
            baseline = conditions.get(token,old)
            if not row['no_op']:
                observations.append(dict(global_step=row['global_step'],native=values[0],cost=cost,
                    old_advantage=cost-old,conditional_advantage=cost-baseline,
                    conditional_history_available=token in conditions))
            conditions[token] = .9*baseline+.1*cost if token in conditions else cost
        memory[key] = .9*old+.1*cost if key in memory else cost
    summary = dict(log=str(options.log),snapshot_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
        measured_steps=len(rows),noop_steps=sum(row['no_op'] for row in rows),
        best_actual=min(row['policy']['current_compression_value'] for row in rows),
        actual_below_minus_five=sum(row['policy']['current_compression_value']<=-5 for row in rows),
        distinct_frames_below_minus_five=len({row['frame'] for row in rows if row['policy']['current_compression_value']<=-5}),
        initial_widths=rows[0]['learned_exploration'],last_widths=rows[-1]['learned_exploration'],
        ablations={},episodes=[])
    for name, batch in [('all',observations),('after_step_500',[r for r in observations if r['global_step']>=500]),
                        ('has_condition_history',[r for r in observations if r['conditional_history_available']])]:
        old = statistics.pvariance(row['old_advantage'] for row in batch)
        new = statistics.pvariance(row['conditional_advantage'] for row in batch)
        summary['ablations'][name] = dict(n=len(batch),old_advantage_variance=old,
            conditional_advantage_variance=new,reduction_percent=100*(1-new/old))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['global_step']//100+1].append(row)
    for episode,batch in sorted(groups.items()):
        summary['episodes'].append(dict(episode=episode,n=len(batch),
            C=statistics.mean(row['policy']['current_compression_value'] for row in batch),
            G=statistics.mean(row['policy']['current_geometry_value'] for row in batch),
            noop=sum(row['no_op'] for row in batch),
            families={family:dict(n=len(selected),C=statistics.mean(r['policy']['current_compression_value'] for r in selected))
                      for family in ('amount_clean_where','where_sampled')
                      for selected in [[r for r in batch if r['credit_family']==family]] if selected},
            replay_reasons=dict(collections.Counter(row['policy']['actual_plan_replay']['reason'] for row in batch))))
    (options.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (options.out/'advantage_ablation.json').write_text(json.dumps(observations,indent=2))
    print(json.dumps({key:summary[key] for key in ('measured_steps','noop_steps','best_actual',
        'actual_below_minus_five','distinct_frames_below_minus_five','ablations')},indent=2))


if __name__=='__main__':
    main()
