"""Two-slot isolated experiment queue; never controls existing processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',required=True,type=Path)
    options=parser.parse_args();root=options.root.resolve()
    logs=root/'late_batch_logs_storage_v2';logs.mkdir(exist_ok=False)
    while not all((root/n/'completed.json').exists() for n in ['cold_before_s22','cold_current_s22']):
        print('[QUEUE] waiting for the two existing diagnostic runs',flush=True);time.sleep(30)
    checkpoint='/data/maejima/log/20261020/MyNetwork_train/pretrained/043722_SparsePCGC/381.pth'
    jobs=[]
    for seed in (21,22):
        for variant in ('current','before'):
            name='mature_'+variant+'_s'+str(seed)+'_storage_v2'
            jobs.append((name,['--variant',variant,'--seed',str(seed),'--steps','48',
                '--checkpoint',checkpoint,'--offset','38100','--validation','4',
                '--probe-steps','2,4,6,8,24,48']))
    for seed in (21,22):
        jobs.append(('proxy_distill_off_s'+str(seed),['--variant','current','--seed',str(seed),
                     '--steps','64','--validation','4','--disable-proxy-distill']))
    jobs += [('unpaired_current_s21',['--variant','current','--seed','21','--views','1','--steps','64','--validation','4']),
             ('ablation_cold_s21',['--variant','current','--evaluate-only',str(root/'cold_current_s21'),
                                  '--steps','64','--validation','4']),
             ('ablation_mature_s21',['--variant','current','--evaluate-only',str(root/'mature_current_s21_storage_v2'),
                                    '--steps','48','--validation','4'])]
    jobs.append(('late_teacher_probe_s21',['--variant','current','--seed','21','--steps','8',
        '--checkpoint',checkpoint,'--offset','38100','--validation','4','--post-window-probe-floor','.05',
        '--probe-steps','2,4,6,8']))
    (logs/'manifest.json').write_text(json.dumps(jobs,indent=2))
    active={};finished=[];last=0.
    while jobs or active:
        while jobs and len(active)<2:
            name,arguments=jobs.pop(0)
            command=[sys.executable,'-B',str(ROOT/'experiments/sparsepcgc_direction/run.py'),
                     '--out',str(root/name),'--seed-cache',str(root/'cold_current_s21/cache'),*arguments]
            stream=(logs/(name+'.txt')).open('x')
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
            active[name]=(process,stream);print('[QUEUE] started',name,process.pid,flush=True)
        for name,(process,stream) in list(active.items()):
            status=process.poll()
            if status is None:continue
            stream.close();finished.append(dict(name=name,exit_code=status));del active[name]
            print('[QUEUE] finished',name,status,flush=True)
            if status:
                # Preserve failure evidence; do not silently replace a failed run.
                jobs=[]
        if time.time()-last>45:
            print('[QUEUE] active',list(active),'remaining',len(jobs),flush=True);last=time.time()
        if active:time.sleep(2)
    (logs/'finished.json').write_text(json.dumps(finished,indent=2))
    if any(r['exit_code'] for r in finished):raise SystemExit(1)


if __name__=='__main__':main()
