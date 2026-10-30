"""Full-plan Actual experiment on a saved, dataset-independent legal pool.

Generic multiscale occupancy features are hypotheses, not Actual bit labels.
All three operations are composed before real native encoding. No training
code, codec source, checkpoint or original log is written.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def occupancy_features(source, pools, scales=(2, 4, 8)):
    result = {}
    for op, candidates in pools.items():
        remove = np.asarray([(c.get('remove_coords') or [[-100, -100, -100]])[0]
                             for c in candidates], dtype=np.int64)
        add = np.asarray([(c.get('add_coords') or [[-100, -100, -100]])[0]
                          for c in candidates], dtype=np.int64)
        columns = []
        for scale in scales:
            cells, counts = np.unique(source//scale, axis=0, return_counts=True)
            low = np.minimum(cells.min(0), -100)
            span = np.maximum(cells.max(0), np.maximum((add//scale).max(0), (remove//scale).max(0)))-low+1
            def keys(x):
                y=x-low
                return (y[:, 0]*span[1]+y[:, 1])*span[2]+y[:, 2]
            order=np.argsort(keys(cells)); occupied=keys(cells)[order]; n=counts[order]
            def lookup(x):
                k=keys(x//scale);i=np.searchsorted(occupied,k);safe=i.clip(max=len(occupied)-1)
                return np.where((i<len(occupied)) & (occupied[safe]==k),n[safe],0)
            nr,na=lookup(remove),lookup(add)
            if op=='Prune':
                gain=(nr==1).astype(float);density=-np.log1p(nr)
            elif op=='Add':
                gain=-(na==0).astype(float);density=np.log1p(na)
            else:
                changed=np.any(remove//scale!=add//scale,axis=1)
                gain=((nr==1).astype(float)-(na==0).astype(float))*changed
                density=np.log1p(na)-np.log1p(nr)
            columns.extend((gain,density))
        result[op]=np.stack(columns,axis=1)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capture',default='001')
    p.add_argument('--random-plans',type=int,default=12)
    p.add_argument('--native-context',type=Path)
    o=p.parse_args();o.out.mkdir(exist_ok=False,parents=True)
    import torch
    from models.utils.loss.actual_encoder import build_actual_encoder
    from experiments.sparsepcgc_direction.common import materialize,physical_points,geometry,geometry_with_fit,digest
    state=torch.load(o.run/('captured_'+o.capture+'.pth'),map_location='cpu')
    context=torch.load(o.run/('geometry_context_'+o.capture+'.pth'),map_location='cpu')
    source=state['source'][0].T.numpy().astype(np.int64)
    args=SimpleNamespace(**json.loads((o.run/'effective_args.json').read_text()))
    args.sparsepcgc_tmp_dir=str(o.out/'tmp');args.sparsepcgc_device='cuda'
    args.sparsepcgc_worker_cpu_trim_interval=1
    args._den6_online_training_step_active=False
    args._sparsepcgc_force_fresh_actual_encode=True
    plan=dict(source=source,pools=state['pools'],**context)
    if o.native_context:
        import time
        from models.utils.pointcloud.autonomous_context_proposals import context_proposals
        start=time.perf_counter()
        state['pools'],diagnostics=context_proposals(source,o.native_context,
            int(np.ceil(len(source)*.06)),31,random_pools=state['pools'])
        plan['pools']=state['pools']
        (o.out/'context_proposals.json').write_text(json.dumps(dict(
            diagnostics=diagnostics,seconds=time.perf_counter()-start),indent=2))
        torch.save(state['pools'],o.out/'context_pools.pth')
    features=occupancy_features(source,state['pools'])
    for op in features:np.save(o.out/(op+'_features.npy'),features[op])
    encoder=build_actual_encoder(args);rng=np.random.RandomState(31);results=[]
    def encode(coords):
        xyz=physical_points(plan,coords)
        stats=encoder.encode_bits(torch.from_numpy(xyz.T.copy()))
        return {k:stats.get(k) for k in ('bit','node','single','point_count','encode_time','sparsepcgc_point_sha256')}
    try:
        baseline=encode(source);repeat=encode(source)
        (o.out/'baseline_repeat.json').write_text(json.dumps(dict(baseline=baseline,repeat=repeat),indent=2))
        profiles={}
        for op,x in features.items():
            profiles[op]={
                'random':rng.normal(size=len(x)),
                'collapse':x[:,0]+2*x[:,2]+4*x[:,4]+.01*x[:,3],
                'density':x[:,1]+x[:,3]+x[:,5]}
            if o.native_context:
                profiles[op]['native_gain']=np.array([c['optimistic_gain_bits'] for c in state['pools'][op]])
            else:
                profiles[op]['learned']=state['arrays']['combined_logits'][op].numpy()
        trials=[]
        for profile in (('native_gain','random') if o.native_context else ('learned','random','collapse','density')):
            for ratio in (.0001,.0003,.001,.003,.01):
                trials.append((profile,{op:ratio for op in profiles}))
        bins=np.array([0.,.0001,.0003,.001,.003,.01,.03])
        for _ in range(o.random_plans):
            trials.append(('native_gain' if o.native_context else 'collapse',
                           {op:float(rng.choice(bins)) for op in profiles}))
        best=None
        for i,(profile,ratios) in enumerate(trials):
            orders={op:np.argsort(-profiles[op][profile],kind='stable').tolist() for op in profiles}
            requested={op:int(np.ceil(len(source)*ratio)) for op,ratio in ratios.items()}
            final,selected=materialize(source,state['pools'],['Add','Prune','Adjust'],requested,orders)
            stats=encode(final);g=geometry(physical_points(plan,source),physical_points(plan,final))
            c=100*(stats['bit']/baseline['bit']-1)
            row=dict(index=i,profile=profile,ratios=ratios,counts={k:len(v) for k,v in selected.items()},
                stats=stats,C=c,geometry=g,final_hash=digest(final))
            results.append(row)
            with (o.out/'plans.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if best is None or c<best['C']:
                best=row
                torch.save(dict(plan=plan,final=final,selected=selected,row=row),o.out/('best_%03d.pth'%i))
        (o.out/'completed.json').write_text(json.dumps(dict(plans=len(results),best=best),indent=2))
    finally:encoder.close()


if __name__=='__main__':main()
