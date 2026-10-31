"""Aggregate measured pipeline runs, retaining failed optimization variants."""
import argparse
import hashlib
import json
from pathlib import Path
import pstats
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.summarize_autonomous_speed import rows, summarize


def measured_run(directory):
    data = rows(directory)
    timing = summarize(directory)
    steady = data[5:]
    walls = timing['wall_seconds'][5:]
    timing['steady_step_start'] = 6
    timing['steady_wall_mean'] = statistics.mean(walls) if walls else None
    timing['steady_wall_median'] = statistics.median(walls) if walls else None
    timing['steady_mean_seconds'] = {
        key: statistics.mean(row['times'].get(key, 0.) for row in steady)
        for key in data[0]['times']} if steady else {}
    timing['steady_native_worker_mean'] = statistics.mean(
        row['native_calls'][-1]['worker']['wall_time_seconds'] for row in steady
        if row.get('native_calls') and 'wall_time_seconds' in row['native_calls'][-1]['worker']) if steady else None
    timing['last_pool_cache'] = data[-1].get('pool_ram_cache')
    timing['source_integrity'] = json.loads((directory/'integrity.json').read_text())
    profile = directory/'cpu_profile.pstats'
    if profile.exists():
        timing['profile_cumulative_seconds'] = {
            name: sum(value[3] for key, value in pstats.Stats(str(profile)).stats.items() if key[2] == name)
            for name in ('clip_joint_gradients', '_build_exact_den6_residual_plan')}
    return timing


def trajectory(before, after):
    old, new = rows(before), rows(after)
    assert [r['step'] for r in old] == [r['step'] for r in new]
    result = []
    for a, b in zip(old, new):
        ac, bc = a['policy']['current_compression_value'], b['policy']['current_compression_value']
        ag, bg = a['policy']['current_geometry_value'], b['policy']['current_geometry_value']
        result.append(dict(step=a['step'], plan_equal=a['plan']==b['plan'],
            counts_equal=a['plan']['selected_counts']==b['plan']['selected_counts'],
            actual_before=ac, actual_after=bc, geometry_before=ag, geometry_after=bg))
    return dict(steps=len(result), equal_plans=sum(x['plan_equal'] for x in result),
        equal_counts=sum(x['counts_equal'] for x in result),
        actual_max_difference=max(abs(x['actual_before']-x['actual_after']) for x in result),
        geometry_max_difference=max(abs(x['geometry_before']-x['geometry_after']) for x in result), rows=result)


def false_measurements(directory):
    log = next((directory/'log').rglob('*.txt')).read_text()
    bits = re.findall(r'StepBits .*?GT=([\d.]+), PolicyMINE=([\d.]+)', log)
    geometry = re.findall(r'StepLoss .*?raw\[geom=([-+\d.eE]+)', log)
    return [dict(step=i+1, gt_bits=float(gt), mine_bits=float(mine),
                 actual_percent=100.*(float(mine)/float(gt)-1.),
                 geometry_log_rounded=float(geometry[i])) for i,(gt,mine) in enumerate(bits)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    names = ('true_before', 'true_after', 'true_after_v2', 'true_after_v3',
             'true_after_v4', 'true_control', 'false_before', 'false_after',
             'light_before', 'light_after', 'false_final', 'full_step_profile')
    result = dict(runs={n:measured_run(args.root/n) for n in names})
    result['final_trajectory'] = trajectory(args.root/'true_before', args.root/'true_after_v4')
    result['unchanged_control_trajectory'] = trajectory(args.root/'true_before', args.root/'true_control')
    result['light_trajectory'] = trajectory(args.root/'light_before', args.root/'light_after')
    result['false_before'] = false_measurements(args.root/'false_before')
    result['false_after'] = false_measurements(args.root/'false_after')
    result['false_measured_equal'] = result['false_before'] == result['false_after']
    result['false_final'] = false_measurements(args.root/'false_final')
    result['false_final_measured_equal'] = result['false_before'] == result['false_final']
    result['fused_native_pilot'] = json.loads((args.root/'fused_native_pilot/summary.json').read_text())
    result['false_geometry_precision'] = 'rounded log, 4 decimals; not a full-precision equality assertion'
    for label, filename in (('false_tensor_comparison','false_comparison.json'),
                            ('true_tensor_comparison','true_comparison_final.json'),
                            ('control_tensor_comparison','true_unchanged_control.json')):
        result[label] = json.loads((args.root/filename).read_text())
    result['false_final_tensor_comparison'] = json.loads((args.root/'false_final_comparison.json').read_text())
    before = json.loads((args.root/'before/manifest.json').read_text())
    result['changed_original_sources'] = [p for p,h in before.items()
        if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h]
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(false_measured_equal=result['false_measured_equal'],
        light_wall_before=result['runs']['light_before']['steady_wall_mean'],
        light_wall_after=result['runs']['light_after']['steady_wall_mean'],
        changed_original_sources=result['changed_original_sources']), indent=2))


if __name__ == '__main__':
    main()
