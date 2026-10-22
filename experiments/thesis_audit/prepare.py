import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    options=parser.parse_args(); out=options.out
    paths=[ROOT/'train.py',ROOT/'test.py',ROOT/'test_BDrate.py',
           *sorted((ROOT/'models').rglob('*.py')),
           *sorted((ROOT.parent/'compress/octree/SparsePCGC').rglob('*.py'))]
    manifest={str(p):sha(p) for p in paths if p.is_file()}
    (out/'protected_sources.json').write_text(json.dumps(manifest,indent=2))
    for cmd,name in [(['git','status','--short'],'start_status.txt'),
                     (['git','diff','--binary'],'start_diff.txt'),
                     (['git','rev-parse','HEAD'],'start_head.txt')]:
        (out/name).write_bytes(subprocess.check_output(cmd,cwd=str(ROOT)))
    source=ROOT/'experiments/results/20261022_actual_direction/mature_current_s21_storage_v2'
    # Exact encoded inputs from a completed checkpoint-validation run, not a
    # replacement point editor or newly selected favourable edits.
    cases=[]
    for index,stem in enumerate(['BlueBackpack_UVG_vox9_25_0_250_0100',
                                 'BlueBackpack_UVG_vox9_25_0_250_0101',
                                 'BlueSpin_UVG_vox9_25_0_250_0100',
                                 'BlueSpin_UVG_vox9_25_0_250_0101']):
        for variant in ['baseline','mine']:
            pattern='validation_000_'+stem+'_encode_*.npz'
            files=sorted(source.glob(pattern))
            if len(files)!=2:
                raise ValueError('Expected baseline and MINE encoded inputs: '+stem)
            path=files[variant=='mine']
            points=np.load(str(path))['points']
            cases.append(dict(id='frame%d_%s'%(index,variant),frame=stem,
                              variant=variant,npz=str(path.resolve()),sha256=sha(path),
                              points=len(points),source_run=str(source.resolve())))
    result=dict(cases=cases,source_options=json.loads((source/'options.json').read_text()),
                effective_args=str((source/'effective_args.json').resolve()),
                validation=str((source/'validation_000.json').resolve()),
                note='Frozen outputs of checkpoint validation, not new Network inference.')
    (out/'cases.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
