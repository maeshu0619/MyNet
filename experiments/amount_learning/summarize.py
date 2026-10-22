"""Read-only aggregation of independent Amount experiments; never fill missing results."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import torch


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def distribution(values):
    positive = [v for v in values if v > 0]
    return dict(count=len(values), mean=statistics.mean(values),
                std=statistics.pstdev(values), minimum=min(values), maximum=max(values),
                minimum_positive=min(positive) if positive else None,
                no_op_count=len(values)-len(positive))


def export(path, records):
    keys = list(dict.fromkeys(key for record in records for key in record))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(records)


def measure(directory, steps, label, gradients, updates, decisions, validations):
    if not (directory/'completed.json').exists():
        raise ValueError('Run has not completed: ' + str(directory))
    info = dict(directory=str(directory), steps=steps,
                integrity=read_json(directory/'integrity.json'))
    if any(info['integrity'].values()):
        raise ValueError('Source/checkpoint integrity failure: ' + str(directory))
    train = [r for r in rows(directory/'decisions.jsonl') if r['phase']=='train'][:steps]
    if len(train) != steps:
        raise ValueError('Incomplete decisions: ' + str(directory))
    ratios = {op: [] for op in ['Add', 'Prune', 'Adjust', 'total']}
    for record in train:
        counts = record['counts']
        total = sum(counts.values())
        ratio = record['debug']['selected_total_ratio']
        changed = record['debug']['selected_changed_voxel_ratio']
        if not (0 <= ratio <= .0099+1e-9 and 0 <= changed <= .0099+1e-9):
            raise ValueError('Safety limit exceeded')
        # Exact builder logs count/N; do not infer the denominator from edited N.
        n = round(total/ratio) if ratio > 0 else None
        values = {op: 100*counts[op]/n if n else 0. for op in counts}
        values['total'] = 100*ratio
        for op, value in values.items():
            ratios[op].append(value)
        decisions.append(dict(run=label, step=record['step']+1,
                              frame=record['frame'], group=record['debug']['factorized_exploration_group'],
                              **{op+'_percent': value for op, value in values.items()},
                              changed_voxel_percent=100*changed))
    info['operation_percent'] = {op: distribution(values) for op, values in ratios.items()}
    info['safety_violations'] = 0
    info['validation'] = {}
    for step in sorted({0, 24, steps}):
        path = directory/('validation_%03d.json' % step)
        if not path.exists():
            raise ValueError('Missing validation: ' + str(path))
        record = read_json(path)
        info['validation'][step] = record
        validations.append(dict(run=label, step=step, C_percent=record['value'],
                                G=record['geometry_value'],
                                validation_objective_logged=record['fixed_objective'],
                                signature=record['sample_signature']))
    grad_records = [r for r in rows(directory/'gradients.jsonl') if r['step'] <= steps]
    for record in grad_records:
        for module, metrics in record['groups'].items():
            gradients.append(dict(run=label, step=record['step'], module=module,
                                  stage=record['stage'], source=record.get('source', ''), **metrics))
    actual = [r for r in grad_records if r['stage']=='source' and r['source']=='actual']
    info['actual_nonzero_probes'] = {
        op: dict(nonzero=sum(r['groups'][op]['l2']>0 for r in actual), probes=len(actual))
        for op in ['AddWhere', 'PruneWhere', 'AdjustWhere', 'AddAmount', 'PruneAmount',
                   'AdjustAmount', 'CoarseAmount']}
    clip = [r for r in grad_records if r['stage']=='clip_model_gradients:after']
    info['global_clip_probes'] = dict(count=len(clip),
        applied=sum(bool(r['debug']['train_grad_clip_applied']) for r in clip))
    # Some old diagnostic versions recorded the auxiliary optimizer too. Last
    # record per update is the decision optimizer, as verified by snapshots.
    update_records = {r['step']: r for r in rows(directory/'updates.jsonl') if r['step']<=steps}
    for step, record in update_records.items():
        for module, metrics in record['update'].items():
            theta = record['theta'][module]['l2']
            updates.append(dict(run=label, step=step, module=module,
                                update_l2=metrics['l2'], update_rms=metrics['rms'],
                                theta_l2=theta, relative_update=metrics['l2']/theta if theta>0 else None,
                                final_gradient_l2=record['gradient'][module]['l2'],
                                final_gradient_rms=record['gradient'][module]['rms']))
    info['training_time'] = read_json(directory/'training_time.json')
    return info, train


def matched(before, after, before_train, after_train):
    old = torch.load(str(before/'model_000.pth'), map_location='cpu')['model']
    new = torch.load(str(after/'model_000.pth'), map_location='cpu')['model']
    # An existing diagnostic balance buffer has a NaN sentinel. Compare bytes
    # (including its NaN payload) rather than falsely treating NaN != NaN as a
    # changed initialization. All trainable parameters must still match exactly.
    changed = [key for key in old if key not in new
               or old[key].shape != new[key].shape or old[key].dtype != new[key].dtype
               or not torch.equal(old[key].reshape(-1).view(torch.uint8),
                                  new[key].reshape(-1).view(torch.uint8))]
    frames_equal = [(r['frame'], r['source_hash']) for r in before_train] == [
        (r['frame'], r['source_hash']) for r in after_train]
    initial_before = read_json(before/'validation_000.json')
    initial_after = read_json(after/'validation_000.json')
    initial_equal = all(initial_before[key]==initial_after[key]
                        for key in ['value', 'geometry_value', 'sample_signature'])
    if changed or not frames_equal or not initial_equal:
        raise ValueError('Unmatched initial state, frames or validation')
    return dict(common_initial_state_tensors_equal=True,
                added_state_keys=sorted(set(new)-set(old)),
                input_frames_and_source_hashes_equal=frames_equal,
                initial_actual_validation_equal=initial_equal)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior-seed22', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    gradients, updates, decisions, validations = [], [], [], []
    summary = dict(runs={}, matched_comparisons={})
    comparisons = [
        ('s21', options.root/'before_s21', options.root/'after_v2_s21', 32),
        ('s22', options.prior_seed22, options.root/'after_v2_s22', 24),
    ]
    baseline_manifest = read_json(options.root/'baseline_source/manifest.json')
    for seed, before, after, steps in comparisons:
        if read_json(before/'source_sha256.json') != baseline_manifest:
            raise ValueError('Baseline source mismatch: ' + str(before))
        train = []
        for variant, path in [('before', before), ('after', after)]:
            label = seed+'_'+variant
            info, records = measure(path, steps, label, gradients, updates, decisions, validations)
            summary['runs'][label] = info
            train.append(records)
        summary['matched_comparisons'][seed] = matched(before, after, *train)
    old_mature = read_json(options.prior_seed22.parent/'mature_current_s21_storage_v2/validation_000.json')
    new_mature = read_json(options.root/'compatibility_v2_legacy_mature/validation_900.json')
    summary['legacy_mature_initial_forward_equal'] = all(old_mature[k]==new_mature[k]
        for k in ['value', 'geometry_value', 'sample_signature'])
    if not summary['legacy_mature_initial_forward_equal']:
        raise ValueError('Legacy checkpoint forward changed')
    summary['legacy_mature_validation'] = new_mature
    summary['limitations'] = [
        'Short independent UVG/m=8 runs, not a full 382-episode performance guarantee.',
        'Actual gradients use the unchanged combined compression/geometry RD teacher, not codec autograd.',
        'Global clip statistics cover probed steps, not uninstrumented steps.',
        'Relative update of initially near-zero Amount weights can be large; use absolute updates too.',
        'Rejected prototype after_s21 and compatibility_legacy_mature are excluded from final comparisons.',
    ]
    (options.out/'summary.json').write_text(json.dumps(summary, indent=2))
    export(options.out/'gradients.csv', gradients)
    export(options.out/'updates.csv', updates)
    export(options.out/'decisions.csv', decisions)
    export(options.out/'validation.csv', validations)
    print(json.dumps(summary, indent=2))


if __name__=='__main__':
    main()
