"""Full fixed-validation Q/RD calibration and causal module rollback results."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.sparsepcgc_direction.common import (
    append_json,canonical_points,correlations,digest,geometry_with_fit,physical_points,write_json,
)
from experiments.sparsepcgc_direction.summarize import export,rows,mean


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--runs',nargs='+',help='Audit only these new runs; preserve earlier audit artifacts')
    opts=parser.parse_args();out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    all_rows=[];calibration=[];ablations=[]
    for run in sorted(opts.root.iterdir()):
        if opts.runs and run.name not in opts.runs:continue
        if not (run/'completed.json').exists() or not (run/'validation_manifest.json').exists():continue
        if run.name.startswith('pilot') or run.name=='teacher_verification':continue
        args=SimpleNamespace(**json.loads((run/'effective_args.json').read_text()))
        encoded=rows(run/'encodes.jsonl');byhash={r['hash']:r['bit'] for r in encoded}
        labels=json.loads((run/'evaluation_labels.json').read_text()) if (run/'evaluation_labels.json').exists() else {}
        saved={};subset=[]
        for file in sorted(run.glob('validation_*_plan.pt')):
            stage=int(file.name.split('_')[1]);plan=torch.load(str(file),map_location='cpu');frame=plan['frame']
            source=digest(canonical_points(physical_points(plan,plan['source'])))
            final=digest(canonical_points(physical_points(plan,plan['final'])))
            assert source in byhash and final in byhash,('Missing native measurement',run.name,file.name)
            base,bit=byhash[source],byhash[final];C=100*(bit/base-1)
            g=geometry_with_fit(plan,plan['final'],plan['selected'],args)
            changed=(plan['counts']['Add']+plan['counts']['Prune']+2*plan['counts']['Adjust'])/len(plan['source'])
            J=args.w_com*C+args.cp_lambda_geom*max(g['training_G_reconstructed']-args.cp_tau_geom,0.)+args.heuristic_guidance_online_edit_tiebreak_weight*changed
            row=dict(run=run.name,stage=stage,label=labels.get(str(stage),str(stage)),frame=frame,
                C=C,G=g['training_G_reconstructed'],J=J,plan_Q=plan['plan_q'],Q_error=plan['plan_q']-J,
                fit=g['fit'],fit_weighted=g['fit_weighted'],chamfer=g['chamfer_sum'],d1_mse=g['d1_mse'],max_distance=g['max_distance'],
                **{op+'_count':plan['counts'][op] for op in plan['counts']},
                amount_bin=plan['debug']['amount_bin_index'],amount_fine=plan['debug']['amount_fine_log_residual'],
                confidence=plan['debug']['network_plan_confidence'],source_hash=source,final_hash=final)
            subset.append(row);all_rows.append(row);saved[(stage,frame)]=plan
            append_json(out/'validation_points.jsonl',row)
        for stage in sorted({r['stage'] for r in subset}):
            group=[r for r in subset if r['stage']==stage]
            observed=json.loads((run/('validation_%03d.json'%stage)).read_text())
            assert abs(mean([r['C'] for r in group])-observed['value'])<1e-8
            assert abs(mean([r['G'] for r in group])-observed['geometry_value'])<2e-6
            calibration.append(dict(run=run.name,stage=stage,label=labels.get(str(stage),str(stage)),n=len(group),
                C=mean([r['C'] for r in group]),G=mean([r['G'] for r in group]),
                Q_RMSE=float(np.sqrt(mean([r['Q_error']**2 for r in group]))),
                Q_MAE=mean([abs(r['Q_error']) for r in group]),
                statistics=correlations([r['plan_Q'] for r in group],[r['J'] for r in group])))
        if labels:
            full=int(next(k for k,v in labels.items() if v=='full'))
            for row in subset:
                control=next(r for r in subset if r['stage']==full and r['frame']==row['frame'])
                a=saved[(full,row['frame'])];b=saved[(row['stage'],row['frame'])]
                selected_a={op:{a['pools'][op][i]['candidate_id'] for i in a['selected'][op]} for op in a['selected']}
                selected_b={op:{b['pools'][op][i]['candidate_id'] for i in b['selected'][op]} for op in b['selected']}
                point_delta=len(set(map(tuple,a['final']))^set(map(tuple,b['final'])))
                feature_change=None
                name=Path(row['frame']).stem
                af=run/('validation_%03d_%s_adapter.npz'%(full,name));bf=run/('validation_%03d_%s_adapter.npz'%(row['stage'],name))
                if af.exists() and bf.exists():
                    x=np.load(str(af))['feature'];y=np.load(str(bf))['feature']
                    feature_change=float(np.linalg.norm((x-y).astype(np.float64))/max(np.linalg.norm(x.astype(np.float64)),1e-30))
                ablations.append(dict(run=run.name,label=row['label'],frame=row['frame'],delta_C_vs_full=row['C']-control['C'],
                    delta_G_vs_full=row['G']-control['G'],point_symmetric_difference=point_delta,
                    adapter_relative_feature_change=feature_change,
                    **{op+'_selection_symmetric_difference':len(selected_a[op]^selected_b[op]) for op in selected_a}))
        print('[VALIDATION AUDIT]',run.name,len(subset),flush=True)
    export(out/'validation_points.csv',all_rows);write_json(out/'plan_critic_calibration.json',calibration)
    export(out/'module_ablations.csv',ablations);write_json(out/'completed.json',dict(runs=len({r['run'] for r in all_rows}),points=len(all_rows)))


if __name__=='__main__':main()
