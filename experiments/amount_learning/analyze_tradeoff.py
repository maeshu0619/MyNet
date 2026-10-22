"""Decompose saved Actual pairs and optimizer directions without training or mutation."""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def export(path, records):
    keys = list(dict.fromkeys(key for record in records for key in record))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    summary, pairs, directions = {}, [], []
    for name in ['before_s21', 'after_v2_s21', 'after_v2_s22']:
        directory = options.root/name
        if not (directory/'completed.json').exists():
            raise ValueError('Incomplete experiment: '+name)
        decisions = {r['step']+1: r for r in rows(directory/'decisions.jsonl') if r['phase']=='train'}
        anchors = {}
        for teacher in rows(directory/'teachers.jsonl'):
            scalars, debug = teacher['scalars'], teacher['debug']
            key = (scalars['input_file'], scalars['factorized_group'], scalars['matched_anchor_hash'])
            if not scalars['matched_reused']:
                anchors[key] = teacher
                continue
            previous = anchors.pop(key, None)
            if previous is None or not debug['actual_group_pair_available']:
                continue
            before = previous['scalars']
            dc = scalars['current_compression_value']-before['current_compression_value']
            dg = scalars['current_geometry_value']-before['current_geometry_value']
            dr = (decisions[teacher['step']]['debug']['selected_changed_voxel_ratio']
                  -decisions[previous['step']]['debug']['selected_changed_voxel_ratio'])
            dj = debug['actor_rd_target']-previous['debug']['actor_rd_target']
            weight_c = scalars['actor_compression_weight']
            weight_g = scalars['actor_geometry_weight']
            remainder = dj-weight_c*dc-weight_g*dg-2*dr
            if abs(remainder)>1e-4:
                raise ValueError('Saved RD teacher cannot be decomposed: '+str(remainder))
            pairs.append(dict(run=name, step=teacher['step'], group=scalars['factorized_group'],
                              delta_C=dc, delta_G=dg, delta_changed_ratio=dr,
                              delta_RD=dj, weighted_C=weight_c*dc, weighted_G=weight_g*dg,
                              tiebreak=2*dr, roundoff=remainder,
                              compression_and_RD_direction_conflict=dc*dj<0))
        for record in rows(directory/'update_directions.jsonl'):
            for value in record['values']:
                if value['source']=='actual' and value['group'] in ['AddAmount', 'PruneAmount', 'AdjustAmount']:
                    if value['descent_update_cosine'] is not None:
                        directions.append(dict(run=name, step=record['step'], **value))
        subset = [r for r in pairs if r['run']==name and r['group'] in ['Amount', 'Fine']]
        ds = [r for r in directions if r['run']==name]
        per_step = {}
        for record in ds:
            per_step.setdefault(record['step'], []).append(record['first_order_loss_change'])
        summary[name] = dict(
            usable_amount_or_fine_pairs=len(subset),
            sampled_variant_RD_better=sum(r['delta_RD']<0 for r in subset),
            sampled_variant_RD_worse=sum(r['delta_RD']>0 for r in subset),
            compression_and_RD_direction_conflicts=sum(r['compression_and_RD_direction_conflict'] for r in subset),
            maximum_RD_identity_roundoff=max((abs(r['roundoff']) for r in subset), default=0),
            measured_nonzero_actual_head_updates=len(ds),
            head_updates_opposing_instantaneous_actual_gradient=sum(r['descent_update_cosine']<0 for r in ds),
            measured_steps_with_nonzero_actual_amount_updates=len(per_step),
            steps_with_total_first_order_actual_ascent=sum(sum(values)>0 for values in per_step.values()))
    summary['interpretation_limits'] = [
        'These are saved training-frame comparisons, not fixed-validation causal rollback tests.',
        'Actual here means the combined RD policy objective, not compression-only autograd.',
        'Adam moments can oppose a single instantaneous head gradient without total-step ascent.',
        'First-order policy loss change does not guarantee improvement of native codec bits.',
        'This analysis cannot identify the dominant cause of validation regression on its own.',
    ]
    export(options.out/'pairs.csv', pairs)
    export(options.out/'update_directions.csv', directions)
    (options.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    main()
