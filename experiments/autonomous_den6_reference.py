"""Replay existing immutable den6 anchors on exactly the held-out frames.

This is a comparator only. No den6 table/rank/plan enters autonomous training.
Missing source anchors fail explicitly; no cache-building worker is invoked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    o=p.parse_args();o.out.mkdir(exist_ok=False,parents=True)
    import torch
    from models.utils.pointcloud.ana_den6_online import _identity,_load_exact_single_plan_teacher
    from plyfile import PlyData
    from models.utils.loss.actual_encoder import build_actual_encoder
    from experiments.sparsepcgc_direction.common import materialize,geometry_with_fit
    args=SimpleNamespace(**json.loads((o.run/'effective_args.json').read_text()))
    args.sparsepcgc_device='cuda';args.sparsepcgc_tmp_dir=str(o.out/'tmp')
    args.sparsepcgc_worker_cpu_trim_interval=1;args._sparsepcgc_force_fresh_actual_encode=True
    loader_args=SimpleNamespace(**vars(args));loader_args.test_route=False
    loader_args.heuristic_guidance_online_cache_dir='/data/maejima/log/mynet_den6_online_cache'
    loader_args.heuristic_guidance_online_compact_reserve_factor=1.
    frames=json.loads((o.run/'final_validation.json').read_text())['sample_names']
    encoder=build_actual_encoder(args);rows=[]
    try:
        for name in frames:
            sequence=name.split('_vox')[0]
            path=Path(args.input_dir)/args.dataname/sequence/name
            if not path.is_file():raise FileNotFoundError(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            teacher=_load_exact_single_plan_teacher(loader_args,_identity(loader_args,path,digest))
            if teacher is None:
                # Older immutable v6 anchors lack the later stratified-pool
                # marker. They are valid comparators after identity and final
                # voxel-hash checks, but are never installed as training pools.
                identity=_identity(loader_args,path,digest)
                cache=Path(loader_args.heuristic_guidance_online_cache_dir)
                native_checkpoint=Path(args.sparsepcgc_ckptdir)
                legacy_checkpoint_id=str(native_checkpoint)+':'+hashlib.sha256(native_checkpoint.read_bytes()).hexdigest()
                for candidate_path in sorted(cache.glob('8i_'+path.stem+'_'+identity['setting_id']+'_*.pt')):
                    candidate=torch.load(str(candidate_path),map_location='cpu')
                    if any(candidate.get(k)!=identity[k] for k in
                           ('input_sha256','setting_id','scale_m','scale_ae','scale_sr')):
                        continue
                    if candidate.get('checkpoint_identifier') not in (identity['checkpoint_identifier'],legacy_checkpoint_id):
                        continue
                    anchor=candidate.get('initial_heuristic_plan',{})
                    if not anchor.get('available') or not anchor.get('final_voxel_hash'):
                        continue
                    teacher=dict(operation_edit_units=candidate['operation_candidate_shortlists'],
                        heuristic_anchor_plan=anchor,operation_priority=candidate['operation_priority'],
                        cache_path=str(candidate_path))
                    break
            if teacher is None:raise RuntimeError('No immutable den6 comparator for '+str(path))
            reference_path=Path(teacher['cache_path'])
            reference_sha=hashlib.sha256(reference_path.read_bytes()).hexdigest()
            pools=teacher['operation_edit_units'];anchor=teacher['heuristic_anchor_plan']
            ids=set(anchor['candidate_ids'])
            orders={op:[i for i,c in enumerate(pool) if c['candidate_id'] in ids] for op,pool in pools.items()}
            counts={op:len(v) for op,v in orders.items()}
            if counts!=anchor['operation_counts']:raise ValueError('Incomplete den6 comparator candidate set')
            raw=PlyData.read(str(path))['vertex'].data
            source=np.unique(np.stack([raw[k] for k in ('x','y','z')],axis=1).astype(np.int64),axis=0)
            priority=anchor.get('metadata',{}).get('operation_order','Add>Prune>Adjust').split('>')
            if set(priority)!=set(pools):priority=list(teacher['operation_priority'])
            final,selected=materialize(source,pools,priority,counts,orders)
            if {op:len(v) for op,v in selected.items()}!=counts:raise ValueError('Den6 reference did not execute completely')
            final_hash=hashlib.sha256(final.astype(np.int64).tobytes()).hexdigest()
            if final_hash!=anchor['final_voxel_hash']:
                raise ValueError('Den6 comparator final voxel hash mismatch')
            def encode(coords):
                return encoder.encode_bits(torch.from_numpy(np.asarray(coords,dtype=np.float32).T.copy()))
            baseline=encode(source);edited=encode(final)
            plan=dict(source=source,pools=pools,voxel_step=1.,voxel_offset=np.zeros(3))
            quality=geometry_with_fit(plan,final,selected,args)
            row=dict(frame=str(path),input_sha256=digest,C=100*(edited['bit']/baseline['bit']-1),
                baseline_bits=baseline['bit'],edited_bits=edited['bit'],geometry=quality,counts=counts,
                reference_cache=str(reference_path),reference_cache_sha256=reference_sha,final_voxel_hash=final_hash)
            if hashlib.sha256(reference_path.read_bytes()).hexdigest()!=reference_sha:
                raise RuntimeError('Immutable reference changed during comparator measurement')
            rows.append(row)
            np.savez_compressed(o.out/(path.stem+'.npz'),source=source,edited=final)
            with (o.out/'measurements.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        (o.out/'summary.json').write_text(json.dumps(dict(count=len(rows),measurements=rows,
            mean_actual=float(np.mean([r['C'] for r in rows])),
            mean_geometry=float(np.mean([r['geometry']['training_G_reconstructed'] for r in rows]))),indent=2))
    finally:encoder.close()


if __name__=='__main__':main()
