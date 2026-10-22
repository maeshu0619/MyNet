"""Feature/NLL association with explicit row identity and scale alignment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyze_native import keys

NAMES=(['x','y','z','abs_x','abs_y','abs_z','radius']+
       ['phase_x','phase_y','phase_z','centered_phase_x','centered_phase_y','centered_phase_z']+
       ['snap_x','snap_y','snap_z','density','curvature','anisotropy']+
       ['quant_merge','quant_density','quant_empty','quant_residual']+
       ['row_exist','mean_occ','single_proxy','self_occ','bit_entropy','sibling_occ','neighbor_occ','child_id']+
       ['cause_node','cause_single','cause_lowprob','cause_context','cause_quant','cause_sparse','cause_outlier','cause_shape']+
       ['sparse_proxy'])


def correlation(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:
        return dict(n=len(x),pearson=None,spearman=None,reason='constant or near-constant input')
    return dict(n=len(x),pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(spearmanr(x,y).correlation))


def export(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(records)


def native_level(directory):
    trace=json.loads((directory/'trace.json').read_text())
    records=[np.load(str(directory/r['array_file'])) for r in trace['stages'] if r['level']==0]
    return {key:np.concatenate([r[key] for r in records]) for key in ['coords','occupancy','nll']}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    options=parser.parse_args();root=options.root
    if not (root/'feature_run/completed.json').exists():raise ValueError('Feature inference incomplete')
    out=root/'feature_analysis';out.mkdir(exist_ok=False)
    index=json.loads((root/'feature_arrays/index.json').read_text())
    args=json.loads((root/'feature_run/effective_args.json').read_text())
    weight_names=['attr_node_weight','attr_single_weight','attr_lowprob_weight','attr_context_weight',
                  'attr_quant_weight','attr_sparse_weight','attr_outlier_weight','attr_shape_weight']
    weights=np.array([args[name] for name in weight_names])
    correlations=[];alignment=[];channel_stats=[]
    for frame,record in enumerate(index):
        arrays=np.load(str(root/'feature_arrays'/record['file']))
        coords=arrays['coords'][0].T.astype(np.int64);xyz=arrays['xyz'][0].T.astype(np.int64)
        features=arrays['features'][0].astype(float)
        slot=lambda q: ((q[:,0]&1)*4+(q[:,1]&1)*2+(q[:,2]&1))/7.
        alignment.append(dict(frame=frame,name=Path(record['frame']).stem,
                              fraction_coordinate_rows_equal=float(np.mean(np.all(coords==xyz,axis=1))),
                              xyz_channels_match_input=float(np.mean(np.all(features[:3].T==xyz,axis=1))),
                              child_id_matches_metadata=float(np.mean(np.isclose(features[30],slot(coords),atol=1e-7))),
                              child_id_matches_input_xyz=float(np.mean(np.isclose(features[30],slot(xyz),atol=1e-7)))))
        for channel,name in enumerate(NAMES):
            v=features[channel]
            channel_stats.append(dict(frame=frame,channel=channel,name=name,minimum=float(v.min()),
                                      maximum=float(v.max()),std=float(v.std()),constant=bool(v.min()==v.max())))
        scores=arrays['causes'][0].astype(float)*weights[:,None]
        scores=scores/np.maximum(scores.sum(axis=0),1e-6)
        values=np.concatenate([features,arrays['occupancy_nll_proxy'][0],scores],axis=0)
        names=NAMES+['occupancy_nll_proxy']+['weighted_score_'+name.removeprefix('cause_') if hasattr(str,'removeprefix')
                                         else 'weighted_score_'+name[6:] for name in NAMES[31:39]]
        baseline=native_level(root/'native_v3'/('frame%d_baseline'%frame))
        edited=native_level(root/'native_v3'/('frame%d_mine'%frame))
        occupied=baseline['occupancy']>0
        _,i0,i1=np.intersect1d(keys(baseline['coords']),keys(edited['coords']),assume_unique=True,return_indices=True)
        new_cost=np.zeros(len(baseline['nll']));new_cost[i0]=edited['nll'][i1]
        saving=baseline['nll']-new_cost
        for basis,point_coords in [('metadata',coords),('input_xyz',xyz)]:
            coarse,inverse=np.unique(point_coords//2,axis=0,return_inverse=True)
            counts=np.bincount(inverse)
            _,ic,ib=np.intersect1d(keys(coarse),keys(baseline['coords'][occupied]),assume_unique=True,return_indices=True)
            if len(ic)!=len(coarse):raise ValueError('Feature-to-coded-voxel mapping incomplete')
            target_cost=baseline['nll'][occupied][ib];target_saving=saving[occupied][ib]
            for name,value in zip(names,values):
                reduced=np.bincount(inverse,weights=value)/counts
                for target,y in [('occupied_voxel_integer_NLL',target_cost),
                                  ('aligned_candidate_NLL_saving_NOT_actual_bits',target_saving)]:
                    correlations.append(dict(frame=frame,basis=basis,feature=name,target=target,
                                             **correlation(reduced[ic],y)))
    export(out/'correlations.csv',correlations);export(out/'row_alignment.csv',alignment)
    export(out/'channel_statistics.csv',channel_stats)
    proxy=[r for r in correlations if r['basis']=='metadata' and r['feature']=='occupancy_nll_proxy'
           and r['target']=='occupied_voxel_integer_NLL']
    summary=dict(row_alignment=alignment,occupancy_nll_proxy_vs_real_NLL=proxy,
                 constant_channels_all_frames=[NAMES[i] for i in range(40)
                                              if all(r['constant'] for r in channel_stats if r['channel']==i)],
                 warning='Association, not causal feature ablation. Mixed coordinate rows must be resolved before importance claims.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
