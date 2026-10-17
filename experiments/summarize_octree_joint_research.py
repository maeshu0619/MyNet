#!/usr/bin/env python
"""Read-only analysis of raw experiment data; writes new summary artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments/results'


def rows(name,file='trials.jsonl'):
    return [json.loads(line) for line in (RESULTS/name/file).read_text().splitlines()]


def main():
    out=RESULTS/'20261017_octree_joint_research'
    out.mkdir(exist_ok=False)
    joint=rows('20261017_octree_joint_masks_v2')
    baselines={r['frame']:r for r in joint if r['trial']=='baseline'}
    best=[r for r in joint if r['trial']=='gpcc_oracle_best']
    if len(baselines)!=4 or len(best)!=8:
        raise AssertionError('Incomplete four-frame / two-budget experiment')
    metrics=[]
    for r in best:
        base=baselines[r['frame']]
        operations=sum(r['counts'].values())
        metrics.append(dict(frame=Path(r['frame']).stem,budget=r['budget'],profile=r['profile'],
            counts=r['counts'],operation_percent=r['operation_percent'],
            gpcc_baseline_bits=base['gpcc']['bits'],gpcc_bits=r['gpcc']['bits'],gpcc_C=r['gpcc_C'],
            oa_baseline_bits=base['oa']['bits'],oa_bits=r['oa']['bits'],oa_C=r['oa_C'],
            baseline_quality=base['quality'],quality=r['quality'],
            joint_rescue_rewrites=r['joint_rescue_rewrites'],
            measured_gpcc_gain_per_operation=(base['gpcc']['bits']-r['gpcc']['bits'])/operations,
            required_gpcc_gain_per_operation_for_50_percent=base['gpcc']['bits']*.5/operations))
    interactions=[]
    for frame in baselines:
        record=next(r for r in best if r['frame']==frame and r['budget']==.0015)
        ablate=[r for r in joint if r['frame']==frame and r['trial']=='ablation']
        single=[r for r in ablate if len(r['operations'])==1]
        baseline=baselines[frame]['gpcc']['bits']
        extra=(baseline-record['gpcc']['bits'])-sum(baseline-r['gpcc']['bits'] for r in single)
        interactions.append(dict(frame=Path(frame).stem,joint_gain_bits=baseline-record['gpcc']['bits'],
            gain_beyond_sum_of_three_isolated_families_bits=extra,
            family_ablations=[dict(operations=r['operations'],bits=r['gpcc']['bits'],
                                   quality=r['quality']) for r in ablate]))
    marginal=[]
    for p in (RESULTS/'20261017_octree_joint_masks_v2').glob('*_proposals.npz'):
        data=np.load(str(p));seq=data['seq']
        leaves=seq[:,-1,1]==seq[:,-1,1].max()
        hist=np.bincount(seq[leaves,-1,0],minlength=256)
        nonzero=hist[hist>0]
        marginal.append(dict(frame=p.stem,leaf_nodes=int(leaves.sum()),
                             empirical_marginal_entropy_bits=float((-nonzero*np.log2(nonzero/nonzero.sum())).sum())))
    # Post-hoc internal transfer check: choose profile only on other sequences.
    # Not a prospective validation, and only four sequences are available.
    profiles=[r['trial'] for r in joint if r['frame']==next(iter(baselines)) and r.get('budget')==.0015 and 'quality_ok' in r]
    cv=[]
    for frame in baselines:
        train=[r for r in joint if r['frame']!=frame and r.get('budget')==.0015 and r['trial'] in profiles]
        choices=[]
        for profile in profiles:
            records=[r for r in train if r['trial']==profile]
            if len(records)==3 and all(r['quality_ok'] for r in records):
                choices.append((float(np.mean([r['gpcc_C'] for r in records])),profile))
        _,profile=min(choices)
        held=next(r for r in joint if r['frame']==frame and r.get('budget')==.0015 and r['trial']==profile)
        cv.append(dict(held_out=Path(frame).stem,profile_selected_on_other_three=profile,
            quality_ok=held['quality_ok'],gpcc_C_after_noop_safety_fallback=held['gpcc_C'] if held['quality_ok'] else 0.,
            note='post-hoc internal cross-validation; not independent prospective network evaluation'))
    summary=dict(joint_result_rows=len(joint),portfolio_proposals=sum('quality_ok' in r for r in joint),
        best=metrics,operation_family_interactions=interactions,marginal_entropy=marginal,
        context_audit=rows('20261017_octree_joint_context','audit.jsonl'),
        actual_search=rows('20261017_octree_joint_actual_longdress')[-1],
        topology=rows('20261017_octree_joint_topology'),
        topology_rescue=rows('20261017_octree_joint_topology_rescue'),
        phase=rows('20261017_octree_joint_phase'),posthoc_internal_profile_cv=cv,
        tracked_diff_exit_code=subprocess.run(['git','diff','--exit-code'],cwd=str(ROOT),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode,
        achieved_50_percent=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    folders=['20261017_octree_joint_masks_v2','20261017_octree_joint_context',
        '20261017_octree_joint_actual_longdress','20261017_octree_joint_topology',
        '20261017_octree_joint_topology_rescue','20261017_octree_joint_phase']
    files=[p for name in folders for p in (RESULTS/name).rglob('*') if p.is_file()]
    files += [ROOT/'experiments'/name for name in ['octree_joint_mask_research.py',
        'octree_joint_phase_research.py','octree_joint_actual_search.py',
        'octree_joint_context_audit.py','octree_joint_topology_research.py',
        'test_octree_joint_mask_research.py','summarize_octree_joint_research.py']]
    files.append(out/'summary.json')
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(summary=str(out/'summary.json'),manifest_files=len(manifest),
        best=[{k:r[k] for k in ['frame','budget','gpcc_C','oa_C','counts']} for r in metrics],
        interactions=[{k:r[k] for k in ['frame','gain_beyond_sum_of_three_isolated_families_bits']} for r in interactions],
        profile_cv=cv),indent=2))


if __name__=='__main__':
    main()
