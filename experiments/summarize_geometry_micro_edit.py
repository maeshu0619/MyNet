#!/usr/bin/env python
"""Persist standalone experiment artifacts and aggregate without training writes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--log')
    p.add_argument('--refresh', action='store_true', help='Refresh only an archive made by this script')
    a = p.parse_args()
    output = Path(a.out)
    if output.exists() and a.refresh and not (output/'manifest.json').is_file():
        raise ValueError('Refusing to refresh a directory without this experiment manifest')
    output.mkdir(parents=True, exist_ok=a.refresh)
    trials, manifest = [], []
    for run in a.runs:
        source = Path(run)
        dest = output/source.name
        # Only explicit standalone result files. Never copy or touch training
        # weights/cache; large npz geometry remains under its original /tmp path.
        dest.mkdir(exist_ok=a.refresh)
        for filename in ('metadata.json','summary.json','trials.jsonl','results.json'):
            file = source/filename
            if file.is_file():
                shutil.copyfile(str(file), str(dest/filename))
                manifest.append(dict(source=str(file), archived=str(dest/filename),
                                     sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        if (source/'trials.jsonl').is_file():
            for line in (source/'trials.jsonl').read_text().splitlines():
                trial = json.loads(line)
                trial['run'] = source.name
                trials.append(trial)
    with (output/'all_trials.csv').open('w', newline='') as file:
        columns = ['run','frame','trial','bits','baseline_bits','compression_percent',
                   'add','prune','adjust','operation_percent','changed_cells_percent',
                   'chamfer_sq','d1_max_direction_mse','max_gt_to_rec_distance',
                   'max_rec_to_gt_distance','codec_sec','sanity_only','prediction']
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trials)
    grouped = defaultdict(list)
    for row in trials:
        grouped[Path(row['frame']).stem].append(row)
    aggregate = dict(trial_rows=len(trials), frames={})
    for key, rows in grouped.items():
        baseline = next(r for r in rows if r['trial']=='baseline')
        eligible = [r for r in rows if not r['sanity_only'] and r['operation_percent']<=.99]
        aggregate['frames'][key] = dict(
            baseline_bits=baseline['bits'], baseline_chamfer=baseline['chamfer_sq'],
            best_micro=min(eligible,key=lambda r:r['bits']),
            best_micro_chamfer_not_worse=min([r for r in eligible if r['chamfer_sq']<=baseline['chamfer_sq']+1e-9],key=lambda r:r['bits']),
            best_micro_no_GT_max_error_increase=min([r for r in eligible if r['max_gt_to_rec_distance']<=baseline['max_gt_to_rec_distance']+1e-9],key=lambda r:r['bits']))
    if a.log:
        # A bounded byte snapshot of an append-only live log; no process change.
        source = Path(a.log)
        size = source.stat().st_size
        with source.open('rb') as file:
            payload = file.read(size)
        text = payload.decode('utf-8',errors='replace')
        values = [float(v) for v in re.findall(r'^Actual Compression Loss: ([+\-\d.eE]+)',text,re.M)]
        snapshots = re.findall(r'^GeometryRDAudit: raw=([\d.eE+-]+), no_edit_reference=([\d.eE+-]+)',text,re.M)
        prefixes = ('GeometryCodecRoute:', 'Active Loss Weights:', 'Surrogate Pretrain/Online Update:',
                    'Compression Surrogate Backward:', 'Surrogate Model is NOT found!!')
        aggregate['live_log_snapshot'] = dict(path=str(source), snapshot_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), observed_actual_steps=len(values),
            negative=sum(v<0 for v in values), zero=sum(v==0 for v in values), positive=sum(v>0 for v in values),
            mean_actual_percent=float(np.mean(values)) if values else None,
            min_actual_percent=min(values) if values else None, max_actual_percent=max(values) if values else None,
            geometry_audit_steps=len(snapshots),
            mean_raw_geometry=float(np.mean([float(v[0]) for v in snapshots])) if snapshots else None,
            mean_no_edit_geometry=float(np.mean([float(v[1]) for v in snapshots])) if snapshots else None,
            configuration_lines=[line for line in text.splitlines() if line.startswith(prefixes)])
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(trial_rows=len(trials), frames={k:dict(best=v['best_micro']['trial'],
        compression_percent=v['best_micro']['compression_percent']) for k,v in aggregate['frames'].items()},
        live_log_snapshot=aggregate.get('live_log_snapshot')),indent=2))


if __name__=='__main__':
    main()
