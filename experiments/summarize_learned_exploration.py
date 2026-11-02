"""Summarize real private training probes, retaining incomplete-run status."""
import argparse
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def avg(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summarize(directory):
    measurements = rows(directory/'steps.jsonl')
    updates = rows(directory/'updates.jsonl')
    sources = rows(directory/'source_gradients.jsonl')
    result = dict(recorded_updates=len(measurements),
        outer_probe_completed=(directory/'probe_conditions.json').is_file(),
        actual_percent=[r['policy'].get('current_compression_value', r['policy'].get('objective'))
                        for r in measurements],
        geometry=[r['policy'].get('current_geometry_value') for r in measurements],
        roles={role:{key:avg(r['modules'][role][key] for r in updates
                            if r['modules'][role][key] is not None)
                     for key in updates[0]['modules'][role]}
               for role in updates[0]['modules']} if updates else {},
        learned_controls_after_update=[r['learned_controls_after_update'] for r in updates],
        source_gradients={},
        maximum_peak_mib=max((r['peak_mib'] for r in measurements),default=None))
    for source in sorted({k for r in sources for k in r['sources']}):
        roles = sorted({k for r in sources for k in r['sources'].get(source,{})})
        result['source_gradients'][source] = {role:dict(
            l2_mean=avg(r['sources'][source][role]['l2'] for r in sources),
            rms_mean=avg(r['sources'][source][role]['rms'] for r in sources
                         if r['sources'][source][role]['rms'] is not None),
            nonzero=sum(r['sources'][source][role]['l2']>0 for r in sources),
            measurements=len(sources)) for role in roles}
    for filename, key in [('final_validation.json','validation'),
                          ('diagnostic_conditions.json','conditions'), ('integrity.json','integrity')]:
        path = directory/filename
        result[key] = json.loads(path.read_text()) if path.is_file() else None
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    options=parser.parse_args()
    result={directory.name:summarize(directory) for directory in sorted(options.root.iterdir())
            if directory.is_dir() and (directory/'steps.jsonl').is_file()}
    with options.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    for name,row in result.items():
        print(name,json.dumps({k:row[k] for k in ('recorded_updates','outer_probe_completed','validation')}))


if __name__=='__main__':
    main()
