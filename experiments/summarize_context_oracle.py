#!/usr/bin/env python
"""Validate and summarize the new independent context-oracle experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import spearmanr
from context_oracle_core import protect_snapshot

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'experiments/results'


def read(name,file='results.jsonl'):
    path=RESULTS/name/file
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean())) if len(a) else dict(n=0)


def main():
    wide=read('20261017_context_oracle_wide_v1')
    bench=read('20261017_context_oracle_bench_v1')
    voxel=read('20261017_context_oracle_voxelization_v1')
    audit=json.loads((RESULTS/'20261017_context_oracle_audit_v1/audit.json').read_text())
    cases=json.loads((RESULTS/'20261017_context_oracle_cases_v1/cases.json').read_text())
    decode=json.loads((RESULTS/'20261017_context_oracle_decode_control_v1/result.json').read_text())
    native_names=['20261017_context_oracle_native_longdress_v1',
        '20261017_context_oracle_native_longdress_v2','20261017_context_oracle_native_redandblack_v1']
    embed_names=['20261017_context_oracle_embedding_longdress_v1','20261017_context_oracle_embedding_redandblack_v1']
    native=[read(name)[-1] for name in native_names]
    embedding=[read(name)[-1] for name in embed_names]
    assert all(x['trial']=='final' for x in native+embedding),'Native / embedding search incomplete'
    native_grid=json.loads((RESULTS/'20261017_context_oracle_native_oa_grid_v1/native_grid_protocol.json').read_text())
    grid_rows=read('20261017_context_oracle_native_oa_grid_v1/study')
    assert len([x for x in grid_rows if x.get('trial')=='same_pool_comparison'])==5
    frames=list(dict.fromkeys(r['frame'] for r in wide if r.get('trial')=='baseline'))
    assert len(frames)==2
    comparisons=[r for r in wide if r.get('trial')=='same_pool_comparison']
    assert len(comparisons)==50,'Wide five-budget/five-selector comparison incomplete'
    table=[]
    for r in comparisons:
        pool=[x for x in comparisons if x['frame']==r['frame'] and x['budget']==r['budget']]
        assert all(x['candidate_plan_ids']==r['candidate_plan_ids'] for x in pool),'Not the same pool'
        plan=next(x for x in wide if x.get('trial')=='full_plan' and x['frame']==r['frame'] and x['id']==r['winner'])
        assert plan['quality_ok']
        ao=next(x for x in pool if x['method']=='actual_oa_oracle')
        ag=next(x for x in pool if x['method']=='actual_gpcc_oracle')
        assert ao['oa_C']<=r['oa_C']+1e-10 and ag['gpcc_C']<=r['gpcc_C']+1e-10
        base=next(x for x in wide if x.get('trial')=='baseline' and x['frame']==r['frame'])
        n=sum(plan['counts'].values())/plan['operation_percent']*100 if plan['operation_percent'] else None
        changed=sum(plan['counts'].values())+plan['counts']['Adjust']
        table.append(dict(frame=Path(r['frame']).stem,budget_percent=100*r['budget'],method=r['method'],
            plan=r['winner'],operation_percent=r['operation_percent'],
            changed_cell_percent=100*changed/n if n else 0.,counts=plan['counts'],
            oa_C=r['oa_C'],gpcc_C=r['gpcc_C'],quality=r['quality'],
            self_gain=plan['edited_node_cdf_gain'],context_gain=plan['unedited_context_cdf_gain'],
            total_gain=plan['total_cdf_gain'],actual_oa_gain=plan['actual_oa_gain'],
            baseline_quality=base['quality']))
    groups=[]
    for frame in frames:
        single=[x for x in bench if x.get('trial')=='candidate' and x['frame']==frame and
                x['operation_count']==1 and 'edits' in x and x['quality_ok']]
        self_good=[x for x in single if x['edited_node_cdf_gain']>0]
        for name,predicate in [('self_good_context_bad',lambda x:x['unedited_context_cdf_gain']<0),
                               ('self_and_context_good',lambda x:x['unedited_context_cdf_gain']>0)]:
            items=[x for x in self_good if predicate(x)]
            groups.append(dict(frame=Path(frame).stem,category=name,n=len(items),
                total_gain=stats([x['total_cdf_gain'] for x in items]),
                context_gain=stats([x['unedited_context_cdf_gain'] for x in items]),
                affected_nodes=stats([x['affected_unedited_cost_nodes'] for x in items]),
                window_fraction=stats([x['window_fraction_mean'] for x in items]),
                mask_population=stats([x['original_mask_population'] for x in items])))
    correlations=[]
    for frame in frames:
        r=[x for x in bench if x.get('trial')=='candidate' and x['frame']==frame and x['quality_ok']]
        correlations.append(dict(frame=Path(frame).stem,n=len(r),
            prior_vs_actual_oa=float(spearmanr([x['current_prior_score'] for x in r],[x['actual_oa_gain'] for x in r])[0]),
            self_vs_actual_oa=float(spearmanr([x['frozen_self_cdf_gain'] for x in r],[x['actual_oa_gain'] for x in r])[0]),
            context_vs_actual_oa=float(spearmanr([x['total_cdf_gain'] for x in r],[x['actual_oa_gain'] for x in r])[0]),
            oa_context_vs_actual_gpcc=float(spearmanr([x['total_cdf_gain'] for x in r],[x['gpcc_gain'] for x in r])[0]),
            gpcc_zero_delta_fraction=float(np.mean([x['gpcc_gain']==0 for x in r]))))
    safety={}
    for name in ('20261017_context_oracle_wide_v1','20261017_context_oracle_bench_v1',
                 '20261017_context_oracle_native_oa_grid_v1/study'):
        before=json.loads((RESULTS/name/'protected_before.json').read_text())
        after=json.loads((RESULTS/name/'protected_after.json').read_text())
        assert before==after,'Protected file hashes differ'
        safety[name]=dict(files=len(before),changed=[])
    latest=protect_snapshot(ROOT)
    assert latest==after,'Protected files changed after benchmark'
    for x in embedding:
        assert x['model_weights_sha256_before']==x['model_weights_sha256_after']
    assert decode['symbol_roundtrip'] and decode['geometry_roundtrip']
    coverage=[]
    for r in table:
        if r['method']!='total_context':
            continue
        native_keys=np.load(str(RESULTS/'20261017_context_oracle_audit_v1'/(r['frame']+'_native_pool.npz')))['keys']
        keys=set(map(tuple,native_keys.tolist()))
        edits=np.load(str(RESULTS/'20261017_context_oracle_wide_v1'/r['frame']/('plan_%03d.npz'%r['plan'])))['edits']
        hit=sum(tuple(x) in keys for x in edits.tolist())
        coverage.append(dict(frame=r['frame'],budget_percent=r['budget_percent'],
            selected_masks=len(edits),native_initial_pool_hits=hit,
            overlap_fraction=hit/len(edits) if len(edits) else 0.))
    selections=[]
    for frame in frames:
        arrays=[x for x in table if x['frame']==Path(frame).stem]
        full=[r for r in wide if r.get('trial')=='full_plan' and r['frame']==frame and r['quality_ok']]
        extra=[r for r in embedding+native if r['frame']==frame]
        extra += [r for r in bench if r.get('trial')=='topology_budget' and r['frame']==frame and r['quality_ok']]
        extra += [r for r in voxel if r.get('trial')=='local_voxelization' and r['frame']==frame and r['quality_ok']]
        best_oa=min([r for r in full+extra if 'oa_C' in r],key=lambda r:r['oa_C'])
        best_gpcc=min(full+extra,key=lambda r:r['gpcc_C'])
        selections.append(dict(frame=Path(frame).stem,best_oa=best_oa,best_gpcc=best_gpcc,
                               observed_30_percent=bool(best_oa['oa_C']<=-30 or best_gpcc['gpcc_C']<=-30)))
    summary=dict(commit=audit['commit'],same_pool_table=table,
        single_context_groups=groups,score_correlations=correlations,
        single_candidate_rows=sum(r.get('trial')=='candidate' for r in bench),
        single_comparisons=[r for r in bench if r.get('trial')=='same_pool_single_comparison'],
        topology_budget=[r for r in bench if r.get('trial')=='topology_budget'],
        patch_audits=[r for r in bench if r.get('trial')=='patch_pool'],
        voxelization=[r for r in voxel if r.get('trial') in ('proposal_audit','local_voxelization')],
        native_search=native,embedding_search=embedding,native_grid_protocol=native_grid,
        native_grid_comparisons=[r for r in grid_rows if r.get('trial')=='same_pool_comparison'],
        cases=cases,implementation_audit=audit,autonomous_small_decode=decode,
        native_initial_pool_coverage=coverage,
        quantization_baseline_control=json.loads((RESULTS/'20261017_context_oracle_baseline_control_v1/results.json').read_text()),
        overall_best=selections,protected_files=safety,
        full_frame_autonomous_oa_decode=False,global_optimum_proven=False)
    out=RESULTS/'20261017_context_oracle_summary_v2';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    header='|Frame|Cap %|Method|Ops %|Changed cells %|OA C %|G-PCC C %|Chamfer|D1 MSE|max GT→rec|max rec→GT|\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    lines=[]
    for x in table:
        g=x['quality']
        lines.append('|%s|%.2f|%s|%.5f|%.5f|%.6f|%.6f|%.9f|%.9f|%.9f|%.9f|' %(
            x['frame'],x['budget_percent'],x['method'],x['operation_percent'],x['changed_cell_percent'],
            x['oa_C'],x['gpcc_C'],g['chamfer_sq'],g['d1_mse'],g['max_gt_to_rec'],g['max_rec_to_gt']))
    (out/'tables.md').write_text(header+'\n'.join(lines)+'\n')
    folders=list(RESULTS.glob('20261017_context_oracle*'))
    files=[p for folder in folders for p in folder.rglob('*') if p.is_file()]
    files += list((ROOT/'experiments').glob('context_oracle*.py'))+[ROOT/'experiments/test_context_oracle.py',Path(__file__).resolve()]
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(set(files))]
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(summary=str(out/'summary.json'),manifest_files=len(manifest),
        best=[dict(frame=x['frame'],oa_C=x['best_oa']['oa_C'],gpcc_C=x['best_gpcc']['gpcc_C']) for x in selections],
        correlations=correlations,protected_files=safety),indent=2))


if __name__=='__main__':
    main()
