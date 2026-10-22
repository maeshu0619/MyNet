"""Read-only integrity checks and manifest for this independent experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import source_hashes,write_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path(sys.argv[1]).resolve()
    current=source_hashes();runs={};checkpoints={}
    for run in sorted(root.iterdir()):
        if not (run/'completed.json').exists() or not (run/'source_sha256.json').exists():continue
        start=json.loads((run/'source_sha256.json').read_text())
        integrity=json.loads((run/'integrity.json').read_text())
        assert start==current and integrity==dict(source_changes=[],checkpoint_changes=[])
        runs[run.name]=dict(core_unchanged=True,integrity=integrity)
        if (run/'checkpoint_sha256.json').exists():checkpoints.update(json.loads((run/'checkpoint_sha256.json').read_text()))
    assert all(sha(Path(p))==value for p,value in checkpoints.items())
    assert not subprocess.check_output(['git','diff','--binary'],cwd=str(ROOT))
    frames=json.loads((root/'cold_current_s21/validation_manifest.json').read_text())
    write_json(root/'final_integrity.json',dict(runs=runs,core_sources=current,
        existing_checkpoint_sha256=checkpoints,validation_raw_file_sha256={p:sha(Path(p)) for p in frames},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=str(ROOT)).decode()))
    excluded={'cache','tmp','__pycache__'};manifest={}
    for directory,dirs,files in os.walk(str(root)):
        dirs[:]=[d for d in dirs if d not in excluded]
        for name in files:
            path=Path(directory)/name
            if path.name=='evidence_manifest.json':continue
            manifest[str(path.relative_to(root))]=sha(path)
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'experiments/sparsepcgc_direction').glob('*')) if p.is_file()}
    write_json(root/'evidence_manifest.json',dict(sha256=manifest,experiment_source_sha256=code,
        excluded_regenerable_directories=sorted(excluded),evidence_file_count=len(manifest)))
    print(json.dumps(dict(successful_runs=len(runs),evidence_files=len(manifest),integrity='PASS')))


if __name__=='__main__':main()
