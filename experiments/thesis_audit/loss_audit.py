"""Re-audit measured gradients only when their training source matches current files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]


def export(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    options=parser.parse_args();out=options.root/'loss_analysis';out.mkdir(exist_ok=False)
    run=ROOT/'experiments/results/20261022_amount_learning/final_default_s21'
    manifest=json.loads((run/'source_sha256.json').read_text())
    differences=[p for p,h in manifest.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if differences:raise ValueError('Cannot call previous gradient measurements current-source evidence: '+str(differences))
    gradients=rows(run/'gradients.jsonl');updates=rows(run/'updates.jsonl')
    flat=[];metrics=[]
    for record in gradients:
        for module,value in record['groups'].items():
            flat.append(dict(step=record['step'],module=module,stage=record['stage'],
                             source=record.get('source',''),**value))
    for module in gradients[0]['groups']:
        record=dict(module=module)
        for source in ['actual','proxy','entropy','L_downstream','weighted_geometry_primary']:
            selected=[r for r in flat if r['module']==module and r['stage']=='source' and r['source']==source]
            record[source+'_max_L2']=max((r['l2'] for r in selected),default=None)
            record[source+'_max_RMS']=max((r['rms'] for r in selected),default=None)
        us=[r['update'][module]['l2'] for r in updates]
        record.update(nonzero_updates=sum(v>0 for v in us),updates=len(us),mean_update_L2=sum(us)/len(us))
        metrics.append(record)
    args=json.loads((run/'effective_args.json').read_text())
    fields=['loss_mode','w_com','cp_lambda_geom','cp_geometry_backward_scale','cp_tau_geom',
            'cp_lambda_nodes','cp_lambda_single','cp_lambda_actuator','cp_lambda_op',
            'cp_lambda_sparsepcgc','cp_full_cloud_actual_correction_weight',
            'compression_primary_aux_target_ratio','compression_primary_aux_balance_min_scale',
            'compression_primary_aux_balance_max_scale','actual_total_bit_objective_mix',
            'direct_octree_structure_targets','w_attr','w_policy','w_actuator','encoder_0grad',
            'heuristic_guidance_online_policy_weight','heuristic_guidance_online_entropy_weight',
            'sparsepcgc_amount_exploration']
    report=dict(source_run=str(run.resolve()),current_training_source_hashes_equal=True,
                measurements='Completed 2026-10-22 independent 8-step run, probes 2/4/6/8; not new training.',
                settings={key:args.get(key) for key in fields},modules=metrics,
                caveat='Source gradients precede policy-only restoration. Actual includes the RD teacher and critic losses.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    export(out/'modules.csv',metrics);export(out/'all_source_and_routing_gradients.csv',flat)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
