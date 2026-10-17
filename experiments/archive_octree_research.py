#!/usr/bin/env python
"""Archive read-only experiment evidence; normalize physical edit accounting."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    folder=Path(a.out)
    folder.mkdir(parents=True,exist_ok=False)
    rows,windows,manifest=[],[],[]
    for source_name in a.runs:
        source=Path(source_name)
        target=folder/source.name
        target.mkdir()
        for file in source.iterdir():
            if file.suffix not in ('.json','.jsonl','.npz'):
                continue
            shutil.copyfile(str(file),str(target/file.name))
            manifest.append(dict(source=str(file),archived=str(target/file.name),
                                 sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        if (source/'trials.jsonl').exists():
            for line in (source/'trials.jsonl').read_text().splitlines():
                row=json.loads(line)
                row['run']=source.name
                details=row['details']
                # Early runs recorded code-space differences for invertible
                # transforms. Do not misreport them as physical voxel edits.
                if details.get('physical_voxel_edits')==0:
                    row['physical_operation_percent']=0.
                    row['net_bits_with_transform_id']=row['bits']+8
                else:
                    row['physical_operation_percent']=(100.*sum(details['counts'].values())/row['baseline_voxels']
                        if 'counts' in details else row['min_operation_percent'])
                    row['net_bits_with_transform_id']=row['bits']
                row['net_compression_percent']=100.*(row['net_bits_with_transform_id']/row['baseline_bits']-1.)
                rows.append(row)
        if (source/'results.jsonl').exists():
            windows.extend(json.loads(line) for line in (source/'results.jsonl').read_text().splitlines())
    columns=['run','frame','codec','trial','bits','baseline_bits','compression_percent',
             'net_compression_percent','physical_operation_percent','chamfer_sq','d1_mse',
             'max_gt_to_rec','max_rec_to_gt','details']
    with (folder/'all_trials.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['codec'],Path(row['frame']).stem)].append(row)
    summary={}
    for (codec,frame),items in grouped.items():
        summary[codec+'|'+frame]=dict(
            baseline=next(r for r in items if r['trial']=='baseline'),
            best_orientation=min([r for r in items if r['trial'].startswith('orientation')]+[
                r for r in items if r['trial']=='baseline'],key=lambda r:r['net_compression_percent']),
            conditional_leaf=[r for r in items if r['trial'].startswith('oa_conditional_leaf')],
            pattern_leaf=[r for r in items if r['trial'].startswith('oa_pattern_leaf')],
            quantization=[r for r in items if r['trial'].startswith('quant_')])
    (folder/'summary.json').write_text(json.dumps(dict(
        raw_trial_rows=len(rows),window_validation_cases=len(windows),frames=summary,
        window_results=windows,aborted_run='octree-oa-leaf-20261017: stopped own process during candidate generation; completed rows only',
        warning='These are independent research/oracle measurements, not trained network results.'),indent=2))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(archived_files=len(manifest),raw_trial_rows=len(rows),
                         window_validation_cases=len(windows),out=str(folder))))


if __name__=='__main__':
    main()
