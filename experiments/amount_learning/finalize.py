"""Check the final opt-in/default runs against their measured source versions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    result = dict(final_runs={})
    for name, reference, enabled in [
        ('final_default_s21', 'before_s21', False),
        ('final_optin_s21', 'after_v2_s21', True),
    ]:
        directory = options.root/name
        assert read(directory/'completed.json')['updates']==8
        assert not any(read(directory/'integrity.json').values())
        assert read(directory/'effective_args.json')['sparsepcgc_amount_exploration']==enabled
        current = [r for r in rows(directory/'decisions.jsonl') if r['phase']=='train']
        old = [r for r in rows(options.root/reference/'decisions.jsonl') if r['phase']=='train'][:8]
        assert len(current)==len(old)==8
        differences = [i+1 for i, (a, b) in enumerate(zip(current, old))
                       if any(a[key]!=b[key] for key in ['frame', 'source_hash', 'final_hash', 'counts'])]
        assert not differences, (name, differences)
        assert all(sha(ROOT/path)==digest
                   for path, digest in read(directory/'source_sha256.json').items())
        first = read(directory/'validation_000.json')
        previous_first = read(options.root/reference/'validation_000.json')
        assert all(first[k]==previous_first[k] for k in ['value', 'geometry_value', 'sample_signature'])
        result['final_runs'][name] = dict(
            source_and_checkpoint_integrity_pass=True, exploration_enabled=enabled,
            first_eight_executed_voxel_hashes_and_counts_match_reference=True,
            reference=reference, validation_000=first,
            validation_008=read(directory/'validation_008.json'))
    baseline = read(options.root/'baseline_source/manifest.json')
    allowed = {'models/modules/structure_actuator.py', 'models/utils/config/args.py'}
    changed = [path for path, digest in baseline.items() if sha(ROOT/path)!=digest]
    assert set(changed)==allowed, changed
    result['modified_original_training_sources'] = changed
    result['other_original_training_sources_unchanged'] = len(baseline)-len(changed)
    command = [sys.executable, '-B', '-m', 'unittest',
               'tests.test_sparsepcgc_amount_policy', 'tests.test_sparsepcgc_credit_fixes',
               'tests.test_sparsepcgc_actual_semantics', 'tests.test_amount_learning_reachability',
               'tests.test_geometry_codec_routes', 'tests.test_geometry_joint_policy']
    tests = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
    (options.out/'tests.txt').write_text(tests.stdout)
    result['tests'] = dict(command=command, returncode=tests.returncode,
                           output_file='tests.txt')
    if tests.returncode:
        raise RuntimeError('Tests failed; see ' + str(options.out/'tests.txt'))
    manifest = {}
    for run in ['before_s21', 'after_v2_s21', 'after_v2_s22',
                'compatibility_v2_legacy_mature', 'final_default_s21', 'final_optin_s21']:
        directory = options.root/run
        for path in sorted(directory.iterdir()):
            if path.is_file() and (path.suffix in {'.json', '.jsonl'}
                                   or (path.name.startswith('model_') and path.suffix=='.pth')):
                manifest[str(path.relative_to(options.root))] = sha(path)
    for path in sorted((options.root/'summary_v3').iterdir()):
        if path.is_file():
            manifest[str(path.relative_to(options.root))] = sha(path)
    (options.out/'checks.json').write_text(json.dumps(result, indent=2))
    (options.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
