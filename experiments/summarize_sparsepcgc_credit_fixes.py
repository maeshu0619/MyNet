"""Export the staged Actual-credit experiments without modifying raw evidence."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def export_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, destination):
    destination.mkdir(parents=True, exist_ok=False)
    source_rows, stages, contributions, cosines, clips, teachers = [], [], [], [], [], []
    summary = {}
    for directory in sorted(root.iterdir()):
        path = directory/'routing.jsonl'
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.open()]
        lookup = {(row['kind'], row.get('step'), row.get('source')): row for row in rows}
        updates = [r for r in rows if r['kind'] == 'optimizer_update']
        conditions = [r for r in rows if r['kind'] == 'policy_conditions']
        name = directory.name
        def get(kind, step, source=None):
            return lookup.get((kind, step, source), {})
        for row in rows:
            step = row.get('step')
            prefix = dict(run=name, step=step)
            if row['kind'] == 'source_gradient':
                for group, values in row['groups'].items():
                    source_rows.append(dict(prefix, source=row['source'], group=group, **values))
            elif row['kind'] == 'optimizer_update':
                for group, values in row['gradient'].items():
                    theta, delta = row['theta'][group]['l2'], row['update'][group]['l2']
                    item = dict(prefix, group=group, optimizer_gradient_l2=values['l2'],
                                optimizer_gradient_rms=values['rms'], theta_l2=theta,
                                update_l2=delta, update_rate=delta/theta if theta else None,
                                small_denominator_warning=theta < 1e-3)
                    for kind, field, label in [('routing','before','joint_backward'),
                                               ('routing','after','after_policy_routing'),
                                               ('balance','after','after_decision_clip'),
                                               ('clip','after','after_global_clip')]:
                        item[label] = get(kind,step).get(field,{}).get(group,{}).get('l2')
                    for source, label in [('L_downstream','compression_pathwise'),
                                          ('weighted_geometry_primary','geometry_pathwise'),
                                          ('policy:actual_compression_policy','actual_rd_policy')]:
                        item[label] = get('source_gradient',step,source).get('groups',{}).get(group,{}).get('l2')
                    stages.append(item)
                for item in row['conditional_adam_loo']:
                    projection = next(r for r in row['source_projection']
                                      if r['group']==item['group'] and r['source']==item['source'])
                    contributions.append(dict(prefix, **item, gradient_l2=projection['l2'],
                                              gradient_rms=projection['rms'],
                                              gradient_projection=projection['gradient_projection']))
                cosines.extend(dict(prefix, **r) for r in row['gradient_cosines'])
            elif row['kind'] == 'balance':
                debug = row['debug']
                for key, value in debug.items():
                    if not key.endswith('_grad_norm_before_balance'):
                        continue
                    stem = key[:-len('_grad_norm_before_balance')]
                    post = debug.get(stem+'_grad_norm_after_balance')
                    clips.append(dict(prefix, group=stem, before=value, after=post,
                        scale=post/value if value and post is not None else None,
                        clipped=debug.get(stem+'_grad_balance_status')=='clipped'))
            elif row['kind'] == 'clip':
                before = row['debug']['train_grad_total_norm_before_clip']
                after = row['after']['ALL']['l2']
                clips.append(dict(prefix,group='global',before=before,after=after,
                    scale=after/before if before else None,
                    clipped=row['debug']['train_grad_clip_applied']))
        for row in conditions:
            d, s = row['debug'], row['scalars']
            teachers.append(dict(run=name,step=row['step'],group=s.get('factorized_group'),
                frame=row.get('input_file'),reused=s.get('matched_reused'),
                pair=d.get('actual_group_pair_available'),delta=d.get('actual_group_pair_delta'),
                actor_pairs=d.get('actual_actor_replay_pairs'),
                direct_candidate_loss=d.get('actual_candidate_critic_replay_weighted'),
                proposal_loss=d.get('actual_proposal_replay_weighted'),
                geometry_auxiliary=d.get('geometry_policy_weighted')))
        n = len(updates)
        frozen = directory.with_name(name+'_source')/'frozen_source_override.txt'
        frozen_integrity = None
        if frozen.is_file():
            original = Path(frozen.read_text().strip())
            manifest = json.loads((original/'manifest.json').read_text())
            frozen_integrity = {file:hashlib.sha256((original/file).read_bytes()).hexdigest()==digest
                                for file,digest in manifest.items()}
        checkpoint_manifest = directory.with_name(name+'_source')/'read_only_checkpoint_sha256.json'
        input_integrity = None
        if checkpoint_manifest.is_file():
            input_integrity = {file:hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
                               for file,digest in json.loads(checkpoint_manifest.read_text()).items()}
        def module_summary(group):
            selected = [r for r in stages if r['run']==name and r['group']==group]
            actual = [r['actual_rd_policy'] for r in selected if r['actual_rd_policy'] is not None]
            return dict(actual_nonzero=sum(v>0 for v in actual),actual_l2_max=max(actual,default=0),
                update_nonzero=sum(r['update_l2']>0 for r in selected),
                update_l2_max=max((r['update_l2'] for r in selected),default=0),
                update_rate_max=max((r['update_rate'] or 0 for r in selected),default=0))
        summary[name] = dict(steps=n,completed=any(r['kind']=='completed' for r in rows),
            strict_pairs=sum(bool(r['debug'].get('actual_group_pair_available')) for r in conditions),
            groups=dict(Counter(r['scalars'].get('factorized_group') for r in conditions)),
            candidate_direct_nonzero=sum(r['debug'].get('actual_candidate_critic_replay_weighted',0)!=0 for r in conditions),
            global_clip_count=sum(r['clipped'] for r in clips if r['run']==name and r['group']=='global'),
            max_adam_clone_error=max((r['max_abs_error'] for r in rows if r['kind']=='adam_clone_validation'),default=None),
            max_decomposition_relative_error=max((r['relative_error'] for r in rows if r['kind']=='decomposition_check'),default=None),
            source_integrity=[r for r in rows if r['kind']=='integrity'],
            frozen_source_verified=frozen_integrity,input_checkpoint_verified=input_integrity,
            modules={g:module_summary(g) for g in [
                'head:den6_candidate_actor','head:den6_candidate_critic','head:den6_candidate_proposal',
                'head:den6_plan_critic','AddWhere','PruneWhere','AdjustWhere',
                'AddAmount','PruneAmount','AdjustAmount','AmountSelector','Gate',
                'module:point_transformer_feature_adapter']})
        validation = directory/'fixed_validation.json'
        if validation.is_file():
            summary[name]['validation'] = json.loads(validation.read_text())
    for name, records in [('source_gradients.csv',source_rows),('routing_stages.csv',stages),
                          ('source_conditional_adam.csv',contributions),('gradient_cosines.csv',cosines),
                          ('clip_events.csv',clips),('actual_teachers.csv',teachers)]:
        export_csv(destination/name, records)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    manifest={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.glob('*/routing.jsonl')}
    (destination/'raw_manifest.json').write_text(json.dumps(manifest,indent=2))
    def number(value):
        return '—' if value is None else '{:.7g}'.format(value)
    def table(header, values):
        return '\n'.join(['| '+' | '.join(header)+' |',
                          '| '+' | '.join(['---']*len(header))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in values])
    focus=['head:den6_candidate_actor','head:den6_candidate_critic',
           'head:den6_candidate_proposal','head:den6_plan_critic',
           'AddWhere','PruneWhere','AdjustWhere','AddAmount','PruneAmount',
           'AdjustAmount','AmountSelector','Gate','module:point_transformer_feature_adapter']
    pages=['# 勾配・更新の数値付録', '',
           'Actualは圧縮単独ではなく、Actual RD教師とCritic教師を含むsource。'
           'Source L2/RMSはrouting前、finalはrouting/decision clip/global clip後。'
           '更新率はΔθ L2 / θ L2。θが0/極小のAmount初期値では比の解釈に注意。', '']
    for run in ('00_baseline24_serial','16_verified_initial','11_mature24_control','17_verified_mature'):
        if run not in summary:
            continue
        pages += ['## '+run,'']
        for step in (2,4,6,8,18,24):
            selected=[r for r in stages if r['run']==run and r['step']==step and r['group'] in focus]
            if not selected:
                continue
            pages += ['### Step '+str(step),'',table(
                ['group','Actual L2','final L2','final RMS','Δθ L2','更新率'],
                [[r['group']]+[number(r[k]) for k in ('actual_rd_policy','optimizer_gradient_l2',
                 'optimizer_gradient_rms','update_l2','update_rate')] for r in selected]),'']
    (destination/'metrics.md').write_text('\n'.join(pages))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.root.resolve(),args.out.resolve())
    print(json.dumps({name:{key:row[key] for key in ('steps','strict_pairs','candidate_direct_nonzero')}
                      for name,row in result.items()},indent=2))
