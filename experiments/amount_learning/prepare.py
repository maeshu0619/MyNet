import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    destination=Path(sys.argv[1]).resolve();destination.mkdir(parents=True,exist_ok=False)
    baseline=destination/'baseline_source';baseline.mkdir()
    paths=[ROOT/'train.py',*sorted((ROOT/'models').rglob('*.py'))]
    manifest={}
    for path in paths:
        if not path.is_file():continue
        relative=path.relative_to(ROOT);target=baseline/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(str(path),str(target));manifest[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (baseline/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for command,name in [(['git','status','--short'],'start_git_status.txt'),
        (['git','diff','--binary'],'start_git_diff.txt'),(['git','rev-parse','HEAD'],'start_git_head.txt')]:
        (destination/name).write_bytes(subprocess.check_output(command,cwd=str(ROOT)))
    print(str(destination),len(manifest))


if __name__=='__main__':main()
