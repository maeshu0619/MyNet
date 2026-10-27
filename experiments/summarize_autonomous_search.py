"""Summarize executed policies and gradients, without reweighting measurements."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.utils.training.sparsepcgc_autonomous_policy import validation_promotion


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def summarize(directory):
    if not (directory/'completed.json').is_file():
        raise ValueError('Incomplete run: ' + str(directory))
    decisions = [row for row in rows(directory/'decisions.jsonl') if row['phase'] == 'train']
    teachers = rows(directory/'teachers.jsonl')
    updates = rows(directory/'updates.jsonl')
    all_gradients = rows(directory/'gradients.jsonl')
    gradients = [row for row in all_gradients
                 if row['stage'] == 'source' and row['source'] == 'actual']
    validation = {p.stem: json.loads(p.read_text()) for p in directory.glob('validation_[0-9][0-9][0-9].json')}
    modules = {}
    for module in updates[0]['update']:
        source = [(row['step'], row['groups'][module]) for row in gradients]
        rates = [row['update'][module]['l2']/row['theta'][module]['l2'] for row in updates
                 if row['theta'][module]['l2'] > 1e-12]
        modules[module] = dict(
            actual_grad_l2_max=max((g['l2'] for _, g in source), default=0.),
            actual_grad_rms_max=max((g['rms'] for _, g in source), default=0.),
            actual_nonzero_probes=sum(g['l2'] > 0 for _, g in source),
            actual_probes=len(source),
            update_l2_max=max(row['update'][module]['l2'] for row in updates),
            update_rms_max=max(row['update'][module]['rms'] for row in updates),
            update_rate_max=max(rates) if rates else None,
            update_rate_undefined_zero_parameter_norm_steps=sum(row['theta'][module]['l2'] <= 1e-12 for row in updates),
            clipped_gradient_l2_max=max((row['groups'][module]['l2'] for row in all_gradients
                                        if row['stage'] == 'clip_model_gradients:after'), default=0.),
            nonzero_updates=sum(row['update'][module]['l2'] > 0 for row in updates))
    counts = {op: [row['counts'][op] for row in decisions] for op in ('Add', 'Prune', 'Adjust')}
    bins = sorted({row['debug']['amount_bin_ratio'] for row in decisions})
    variants = {}
    for row in decisions:
        variants.setdefault(row['frame'], set()).add(row['final_hash'])
    current = [row['scalars'] for row in teachers]
    return dict(updates=len(updates), initial_weight_digest=json.loads((directory/'initial_weights.json').read_text()),
                integrity=json.loads((directory/'integrity.json').read_text()), validation=validation,
                amount_bins_actually_executed=bins,
                count_ranges={op: [min(v), max(v)] for op, v in counts.items()},
                distinct_executed_clouds_per_frame={Path(k).name: len(v) for k, v in variants.items()},
                strict_actual_pairs=sum(bool(row['debug'].get('actual_group_pair_available', False)) for row in teachers),
                training_actual_mean=sum(row['current_compression_value'] for row in current)/len(current),
                training_geometry_mean=sum(row['current_geometry_value'] for row in current)/len(current),
                modules=modules)


def measured_validation(directory, clock):
    phase = 'validation_%03d' % clock
    encoded = rows(directory/'encodes.jsonl')
    available = sorted({row['phase'] for row in encoded if row['phase'].startswith('validation_')})
    if phase not in available and len(available) == 1:
        phase = available[0]
    gt, mine = {}, {}
    for row in encoded:
        if row['phase'] == 'validation_000' and row['frame'] not in gt:
            gt[row['frame']] = row['bit']
        if row['phase'] == phase:
            if row['frame'] not in gt:
                gt[row['frame']] = row['bit']
            mine[row['frame']] = row['bit']
    measured = []
    seen = {}
    for row in rows(directory/'validation_geometry.jsonl'):
        if row['clock'] != clock or row['frame'] not in mine or row['frame'] not in gt:
            continue
        result = dict(frame=row['frame'], codec_setting=row['codec_setting'],
                             input_sha256=row['input_sha256'], geometry_setting=row['geometry_setting'],
                             actual_percent=100.*(mine[row['frame']]/gt[row['frame']]-1.),
                             geometry=row['geometry'])
        if row['frame'] in seen:
            if seen[row['frame']] != result:
                raise ValueError('Repeated measurement differs; cannot collapse duplicates')
            continue
        seen[row['frame']] = result
        measured.append(result)
    return measured


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    summary = {name: summarize(options.root/name) for name in options.runs}
    (options.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: {k: v for k, v in values.items() if k != 'modules'}
                      for name, values in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
