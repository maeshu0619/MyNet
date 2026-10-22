"""Four-frame descriptive associations. Never replicate one frame's bit label per voxel."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    o=p.parse_args();out=o.root/'plan_correlations';out.mkdir(exist_ok=False)
    native=json.loads((o.root/'native_analysis/summary.json').read_text())['frames']
    args=json.loads((o.root/'feature_run/effective_args.json').read_text())
    names=['node','single','lowprob','context','quant','sparse','outlier','shape']
    weights=np.array([args['attr_'+name+'_weight'] for name in names])
    scores=[]
    for i in range(4):
        a=np.load(str(o.root/'feature_arrays'/('%03d.npz'%i)))['causes'][0].astype(float)
        a=a*weights[:,None];a=a/np.maximum(a.sum(0),1e-6)
        scores.append(a.mean(1))
    scores=np.array(scores)
    result=[]
    for j,name in enumerate(names):
        for target,y in [('actual_saved_bits',[-r['delta_bits'] for r in native]),
                         ('actual_rate_improvement_percent',[-r['physical_rate_change_percent'] for r in native])]:
            x=scores[:,j];y=np.array(y)
            result.append(dict(score='point_mean_'+name,target=target,n_frames=4,
                               pearson=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 else None,
                               spearman=float(spearmanr(x,y).correlation) if x.std()>1e-12 else None))
    report=dict(results=result,warning=(
        'Only four frames from two sequences. Frame-mean scores, not individual-candidate causal values. '
        'Sequence/content and point count confound these associations; no significance or importance claim.'))
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
