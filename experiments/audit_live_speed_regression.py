"""Read-only before/current log comparison; no training, no GPU allocation."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def stats(rows):
    if not rows:
        return dict(n=0)
    step = [r['step_before_audit'] for r in rows]
    actual = [r['actual_seconds'] for r in rows]
    other = [a-b for a,b in zip(step,actual)]
    return dict(n=len(rows),first_step=rows[0]['global_step'],last_step=rows[-1]['global_step'],
        step_mean=statistics.mean(step),step_median=statistics.median(step),
        step_p90=sorted(step)[min(len(step)-1,int(.9*len(step)))],
        actual_mean=statistics.mean(actual),other_mean=statistics.mean(other),
        edit_count_mean=statistics.mean(sum(r['counts'].values()) for r in rows),
        gt_encodes=sum(bool((r.get('context') or {}).get('gt_encoded')) for r in rows),
        noop=sum(bool(r['no_op']) for r in rows),
        C_mean=statistics.mean(r['policy']['current_compression_value'] for r in rows))


def read_log(path, folder):
    rows, events, settings, probes = [], [], [], set()
    period = 'before_cpu_fallback'
    markers = ('KNNBackend:','OptimizerGroups:','StreamingExactCachePolicy:',
        'TrainingPerformanceContract:','TrainingRoute:','AutonomousExploration:',
        'ExplorationSchedule:','Runtime Tradeoff:','SparsePCGC Teacher:',
        'FixedEpisodeFrames:','Gradient Diagnostics:','Compression Teacher Refresh:')
    for number,line in enumerate(path.open(),1):
        if line.startswith(markers) and len(settings)<40:
            settings.append(line.strip())
        if line.startswith(('SparsePCGCAutoCPUFallback:','SparsePCGCHostAdmissionWait:',
                            'SparsePCGCGPUAdmissionWait:')):
            events.append(dict(line=number,after_step=rows[-1]['global_step'] if rows else None,text=line.strip()))
            if line.startswith('SparsePCGCAutoCPUFallback:'):
                period = 'after_cpu_fallback'
        if line.startswith('Den6SourceGradProbe: '):
            match = re.match(r'Den6SourceGradProbe: step=(\d+)',line)
            if match:
                probes.add(int(match[1])-1)
        if line.startswith('AutonomousLearningAudit: '):
            try:
                row = json.loads(line.split(': ',1)[1])
            except ValueError:
                continue
            row['_device_period'] = period
            rows.append(row)
    folder.mkdir(parents=True,exist_ok=False)
    snapshot = ''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)
    (folder/'audit_snapshot.jsonl').write_text(snapshot)
    (folder/'events.json').write_text(json.dumps(events,indent=2))
    (folder/'settings.txt').write_text('\n'.join(settings)+'\n')
    return rows, events, probes, hashlib.sha256(snapshot.encode()).hexdigest()


def csv_intervals(path, window, allowed_steps=None):
    rows = list(csv.DictReader(path.open()))
    starts = {int(r['global_step']):r for r in rows if r['phase']=='step_after_data_load'}
    ends = {int(r['global_step']):r for r in rows if r['phase']=='step_after_cleanup'}
    full, gaps = [], []
    for step in sorted(ends):
        if allowed_steps is not None and step not in allowed_steps:
            continue
        if step in starts:
            full.append(dict(step=step,seconds=float(ends[step]['unix_time'])-float(starts[step]['unix_time'])))
        if step+1 in starts and ends[step]['episode']==starts[step+1]['episode']:
            gaps.append(dict(step=step+1,seconds=float(starts[step+1]['unix_time'])-float(ends[step]['unix_time'])))
    return dict(full_tail_n=len(full[-window:]),
                full_tail_mean=statistics.mean(r['seconds'] for r in full[-window:]) if full else None,
                outside_step_gap_n=len(gaps[-window:]),
                outside_step_gap_mean=statistics.mean(r['seconds'] for r in gaps[-window:]) if gaps else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True,exist_ok=False)
    old,old_events,old_probes,old_sha = read_log(options.before,options.out/'before')
    now,events,probes,now_sha = read_log(options.current,options.out/'current')
    last = now[-100:];first,last_step = last[0]['global_step'],last[-1]['global_step']
    matched_old = [r for r in old if first<=r['global_step']<=last_step]
    matched_by = {r['global_step']:r for r in matched_old}
    frame_equal = all(r['global_step'] in matched_by and r['frame']==matched_by[r['global_step']]['frame'] for r in last)
    result = dict(before_log=str(options.before),current_log=str(options.current),
        snapshot_sha256={'before':old_sha,'current':now_sha},
        measured_steps={'before':len(old),'current':len(now)},
        current_tail=stats(last),before_matched_global_steps=stats(matched_old),matched_frames_equal=frame_equal,
        before_matching_training_span=stats([r for r in old if 100<=r['global_step']<=now[-1]['global_step']]),
        current_matching_training_span=stats([r for r in now if r['global_step']>=100]),
        current_before_cpu_fallback=stats([r for r in now if r['_device_period']=='before_cpu_fallback'][-100:]),
        current_after_cpu_fallback=stats([r for r in now if r['_device_period']=='after_cpu_fallback'][-100:]),
        current_probe_tail=stats([r for r in now[-300:] if r['global_step'] in probes]),
        current_no_probe_tail=stats([r for r in now[-300:] if r['global_step'] not in probes]),
        old_events=old_events,current_events=events,
        csv_before=csv_intervals(options.before.with_name(options.before.stem+'_memory_diagnostics.csv'),100),
        csv_current=csv_intervals(options.current.with_name(options.current.stem+'_memory_diagnostics.csv'),100),
        csv_before_matched=csv_intervals(options.before.with_name(options.before.stem+'_memory_diagnostics.csv'),100,set(range(first,last_step+1))),
        csv_current_matched=csv_intervals(options.current.with_name(options.current.stem+'_memory_diagnostics.csv'),100,set(range(first,last_step+1))),
        blocks=[])
    groups = collections.defaultdict(list)
    for r in now:
        groups[r['global_step']//100].append(r)
    for block,rs in sorted(groups.items()):
        lo,hi=rs[0]['global_step'],rs[-1]['global_step']
        result['blocks'].append(dict(block=block,current=stats(rs),before=stats([r for r in old if lo<=r['global_step']<=hi])))
    (options.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:result[key] for key in ('measured_steps','current_tail','before_matched_global_steps',
        'matched_frames_equal','current_before_cpu_fallback','csv_current','current_events')},indent=2))


if __name__=='__main__':
    main()
