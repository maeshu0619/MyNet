"""Read existing training evidence; never run or modify the training process.

Separates sampled behavior, deterministic validation, and measured-plan
self-imitation. Conditional groups are observational, not causal ablations.
All outputs must go into a new directory.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    protected = [root / 'train.py', root / 'test.py', *sorted((root / 'models').rglob('*.py'))]
    protected = [p for p in protected if p.is_file()]
    before = {str(p): sha(p) for p in protected}
    inputs = [options.log, *[options.log.with_name(options.log.stem + suffix) for suffix in
              ('_epi_metrics.csv', '_epi_point_edits.csv', '_checkpoint_metrics_epi.csv', '_training_frames.json')]]
    input_hashes = {str(p): sha(p) for p in inputs}
    manifest = json.loads(inputs[-1].read_text())
    steps_per_episode = int(manifest['unique_frames_per_episode'])
    metrics, edits, validations = [
        {int(r['episode']): r for r in csv.DictReader(path.open())} for path in inputs[1:4]]
    rows, probes, schedule = [], collections.defaultdict(list), None
    frames = collections.defaultdict(list)
    with options.log.open() as stream:
        for line in stream:
            if line.startswith('ExplorationSchedule: '):
                schedule = line.strip()
            elif line.startswith('AutonomousLearningAudit: '):
                row = json.loads(line.split(': ', 1)[1])
                rows.append(row)
                frames[row['global_step'] // steps_per_episode + 1].append(row['frame'])
            elif line.startswith('Den6SourceGradProbe: '):
                match = re.match(r'Den6SourceGradProbe: step=(\d+), norms=(.*)', line)
                if match:
                    probes[(int(match[1])-1)//steps_per_episode+1].append(ast.literal_eval(match[2]))
    assert len({r['global_step'] for r in rows}) == len(rows), 'duplicate audit steps'
    epochs = collections.defaultdict(list)
    for row in rows:
        epochs[row['global_step']//steps_per_episode+1].append(row)
    records = []
    for ep, batch in sorted(epochs.items()):
        conditional = collections.defaultdict(list)
        for row in batch:
            coefficients = list(row.get('structured_where', {}).values())
            assert not coefficients or len(set(coefficients)) == 1, 'not a shared coefficient'
            key = ('where_clean' if not coefficients else
                   'where_coefficient_negative' if coefficients[0] < 0 else 'where_coefficient_nonnegative')
            conditional[key].append(row)
        replay = [r['policy']['actual_plan_replay'] for r in batch]
        record = dict(episode=ep, steps=len(batch), complete=len(batch)==steps_per_episode,
            actual_percent=mean(r['policy']['current_compression_value'] for r in batch),
            geometry=mean(r['policy']['current_geometry_value'] for r in batch),
            phase_min=min(r['exploration'] for r in batch), phase_max=max(r['exploration'] for r in batch),
            preference_mean=mean(r['native_context_preference'] for r in batch),
            amount_lp_zero=sum(r['amount_log_prob']==0 for r in batch),
            where_lp_zero=sum(r['where_log_prob']==0 for r in batch),
            replay_used=sum(r['used'] for r in replay),
            new_profitable_records=sum(r['stored_new_profitable_plan'] for r in replay),
            replay_reasons=dict(collections.Counter(r['reason'] for r in replay)),
            amount_bin_counts={op:dict(collections.Counter(str(round(r['amount_bins'][op], 7))
                for r in batch)) for op in ('Add', 'Prune', 'Adjust')},
            ratios_mean={op:mean(r['ratios'][op] for r in batch) for op in ('Add', 'Prune', 'Adjust')},
            conditional_groups={key:dict(n=len(group),
                actual_percent=mean(r['policy']['current_compression_value'] for r in group),
                geometry=mean(r['policy']['current_geometry_value'] for r in group))
                for key,group in conditional.items()},
            source_gradient_probe_n=len(probes[ep]),
            source_gradient_l2_mean={key:mean(r.get(key, 0.) for r in probes[ep])
                for key in sorted({key for probe in probes[ep] for key in probe})})
        if ep in metrics and record['complete']:
            assert abs(record['actual_percent']-float(metrics[ep]['actual_compression'])) < 1e-7
        if ep in validations:
            v = validations[ep]
            record.update(validation_actual_percent=float(v['full_cloud_val_actual_percent']),
                validation_geometry=float(v['full_cloud_val_geometry']),
                validation_signature=v['full_cloud_val_sample_signature'],
                optimizer_steps=int(v['optimizer_step_count']),
                geometry_reference=float(v['fixed_validation_geom_reference']))
        records.append(record)
    weights = []
    if options.checkpoint_root:
        import torch
        for ep in (1, 40, 80, 89, 90, 95, 96, 100, 130, 167):
            path = options.checkpoint_root / (str(ep)+'.pth')
            if not path.is_file():
                continue
            original_sha = sha(path)
            state = torch.load(path, map_location='cpu')
            parameter = 'actuator.den6_candidate_actor.native_context_preference'
            weights.append(dict(episode=ep, checkpoint=str(path), sha256=original_sha,
                                native_context_preference=float(state[parameter])))
            assert sha(path) == original_sha
    result = dict(schedule=schedule, training_steps=len(rows), complete_episodes=sum(r['complete'] for r in records),
        first_zero_exploration_step=next((r['global_step'] for r in rows if r['exploration']==0), None),
        same_training_frame_order=all(v==frames[1] for v in frames.values() if len(v)==steps_per_episode),
        checkpoints=weights, episodes=records,
        limitations=['Existing Actual measurements only; no new codec encode or training was run.',
                     'Conditional groups differ in actions and frames; not a fixed-weight causal ablation.',
                     'Source gradients are pre-clip probe norms, not optimizer update norms.'])
    (options.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full = [r for r in records if r['complete'] and 'validation_actual_percent' in r]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    x = [r['episode'] for r in full]
    axes[0].plot(x, [r['actual_percent'] for r in full], label='Train: sampled Actual C [%]')
    axes[0].plot(x, [r['validation_actual_percent'] for r in full], label='Fixed validation: greedy Actual C [%]')
    axes[0].set_ylabel('Actual change [%]'); axes[0].legend()
    axes[1].plot(x, [r['phase_max'] for r in full], label='Exploration multiplier (max per episode)')
    axes[1].plot(x, [r['conditional_groups'].get('where_coefficient_negative',{}).get('n',0)/r['steps'] for r in full], label='Negative Where coefficient fraction')
    axes[1].legend(); axes[1].set_ylim(-.02, 1.02)
    axes[2].plot(x, [r['geometry'] for r in full], label='Train geometry')
    axes[2].plot(x, [r['validation_geometry'] for r in full], label='Fixed validation geometry')
    axes[2].legend(); axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Geometry loss')
    for axis in axes:
        axis.axvline(95.5, color='gray', linestyle=':'); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(options.out/'annealing_audit.png', dpi=160); plt.close(fig)
    changed = [p for p,h in before.items() if sha(Path(p)) != h]
    changed_inputs = [p for p,h in input_hashes.items() if sha(Path(p)) != h]
    integrity = dict(source_changes=changed, input_changes=changed_inputs,
                     source_sha256=before, input_sha256=input_hashes)
    (options.out/'integrity.json').write_text(json.dumps(integrity, indent=2))
    assert not changed and not changed_inputs
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','checkpoints')}, indent=2))


if __name__ == '__main__':
    main()
