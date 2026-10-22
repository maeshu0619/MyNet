"""Verify immutable inputs, run tests, and fingerprint the final evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from utils.bdrate.curves import bd_rate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    o=parser.parse_args();out=o.root/'verification_v2';out.mkdir(exist_ok=False)
    manifest=json.loads((o.root/'protected_sources.json').read_text())
    changed=[p for p,h in manifest.items() if sha(Path(p))!=h]
    if changed:raise ValueError('Protected sources changed: '+str(changed))
    cases=json.loads((o.root/'cases.json').read_text())
    for c in cases['cases']:
        if sha(Path(c['npz']))!=c['sha256']:raise ValueError('Saved input changed')
    source=Path(cases['cases'][0]['source_run'])
    previous=[json.loads(line) for line in (source/'encodes.jsonl').read_text().splitlines()
              if json.loads(line)['phase']=='validation_000']
    rates=[]
    for case in cases['cases']:
        t=json.loads((o.root/'native_v3'/case['id']/'trace.json').read_text())
        main=t['stream_measurement'][1][0]['bytes']*8
        decoder_side=4*(t['codec']['scale_sr']+t['codec']['scale_ae'])
        logical=main+decoder_side
        # Current worker's compatibility wrapper takes normalized file_size,
        # already logical in encoder_multiple, then adds the count bits AGAIN.
        # Record this discrepancy; never change physical RD measurements to fit.
        expected_teacher=logical+decoder_side
        matches=[r for r in previous if Path(r['frame']).stem==case['frame'] and r['n']==case['points']]
        if len(matches)!=1 or matches[0]['bit']!=expected_teacher:
            raise ValueError('Native rate differs beyond the identified teacher-side accounting offset')
        rates.append(dict(case=case['id'],saved_logical_bits=matches[0]['bit'],
                          physical_main_bits=main,native_coder_logical_bits=logical,
                          teacher_duplicate_count_bits=decoder_side,
                          saved_teacher_accounting_reproduced=True))
    checkpoint=cases['source_options']['checkpoint']
    old_hashes=json.loads((source/'checkpoint_sha256.json').read_text())
    for p,h in old_hashes.items():
        if sha(Path(p))!=h:raise ValueError('Original checkpoint or sidecar changed')
    measurements=[json.loads(p.read_text()) for p in sorted((o.root/'rd_frame0').glob('*_*/measurement.json'))]
    series={s:sorted([r for r in measurements if r['series']==s],key=lambda r:r['rate_index'])
            for s in ['Baseline','MINE']}
    sensitivity={}
    for method in ['akima','pchip']:
        sensitivity[method]=bd_rate([r['bpp'] for r in series['Baseline']],
            [r['d1_psnr'] for r in series['Baseline']], [r['bpp'] for r in series['MINE']],
            [r['d1_psnr'] for r in series['MINE']],method=method)
    command=[sys.executable,'-B','-m','unittest','experiments.thesis_audit.test_accounting',
             'tests.test_bdrate_evaluation']
    result=subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          universal_newlines=True)
    (out/'tests.txt').write_text(result.stdout)
    if result.returncode:raise RuntimeError('Tests failed')
    summary=dict(protected_source_files=len(manifest),protected_source_changes=changed,
                 original_checkpoint_hashes=old_hashes,native_bits_reproduction=rates,
                 actual_RD_measurements=len(measurements),tests_command=command,
                 tests_returncode=result.returncode,interpolation_sensitivity=sensitivity,
                 scope='4 held-out frames from 2 UVG sequences; 1-frame six-RD-point frozen-output replay.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    paths=[]
    for folder in ['native_v3','native_analysis','feature_arrays','feature_analysis','loss_analysis',
                   'rd_frame0','plan_correlations','verification']:
        paths.extend(p for p in (o.root/folder).rglob('*') if p.is_file())
    artifacts={str(p.relative_to(o.root)):sha(p) for p in sorted(paths)}
    (out/'manifest.json').write_text(json.dumps(artifacts,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
