#!/usr/bin/env python
"""Read-only implementation audit + cold proposal coverage; new files only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from context_oracle_core import load_ply,unique,protect_snapshot
from models.utils.pointcloud.geometry_joint_proposals import attach_joint_geometry_guidance


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--frames',nargs='+',required=True);a=p.parse_args()
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];before=protect_snapshot(root)
    binary=root.parent/'compress/octree/G-PCC/build/tmc3/tmc3'
    help_result=subprocess.run([str(binary),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    help_text=help_result.stdout+help_result.stderr
    (folder/'gpcc_help.txt').write_text(help_text)
    audit=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(root),text=True).strip(),
        help_returncode=help_result.returncode,
        probability_help_lines=[line for line in help_text.splitlines() if any(x in line.lower() for x in ('probab','trace','dump','entropy'))],
        native_per_node_probability_api=False,
        probability_evidence=['tmc3/geometry_octree_encoder.cpp: OBUF context evolution + inferred bits',
            'tmc3/entropydirac.h: threshold-clamped private probability before Schro bit coding',
            'existing GPCCGeometryEncoder exports stream size, not per-node models'],
        effective_oa_actual_mode='geometry_codec.configure_geometry_codec sets True unless explicitly overridden',
        comparison_current_score='zero-Actor default empirical prior; not running Network checkpoint evaluation',
        proposal_initialization=[])
    for frame in a.frames:
        raw=load_ply(frame,return_color=False).astype(np.float64)
        absolute=unique(np.rint(raw/2.));q=absolute-absolute.min(0)
        args=SimpleNamespace(compress='G-PCC',qs=2,gpcc_effective_qs=2.,gpcc_prequantize=True)
        context={'full_global_voxel_coords':torch.from_numpy(q.T.copy())[None]}
        out=attach_joint_geometry_guidance(context,args,device=torch.device('cpu'))
        payload=out['ana_den6_ranked_candidate_guidance'];pools=payload['operation_edit_units']
        data=np.load(str(root/'experiments/results/20261017_octree_joint_masks_v2'/(Path(frame).stem+'_proposals.npz')))
        seq=data['seq'];leaf=np.flatnonzero(seq[:,-1,1]==seq[:,-1,1].max())
        by_parent={tuple(seq[r,-1,3:6]//2):int(r) for r in leaf}
        legal={(int(x['row']),int(x['mask'])) for x in data['candidates']}
        keys=set();raw_candidates=0;single_parent=0
        for op,pool in pools.items():
            for candidate in pool:
                raw_candidates+=1
                rem=candidate['remove_coords'];add=candidate['add_coords']
                parents={tuple(np.asarray(point)//2) for point in rem+add}
                if len(parents)!=1 or next(iter(parents)) not in by_parent:
                    continue
                row=by_parent[next(iter(parents))];after=int(seq[row,-1,0])
                for points,adding in ((rem,False),(add,True)):
                    for point in points:
                        x,y,z=np.asarray(point)%2;bit=1<<int(4*x+2*y+z)
                        after=after|bit if adding else after&~bit
                if after:
                    single_parent+=1;keys.add((row,after))
        supported=keys&legal
        np.savez_compressed(folder/(Path(frame).stem+'_native_pool.npz'),
                            keys=np.array(sorted(keys),dtype=np.int64).reshape(-1,2))
        audit['proposal_initialization'].append(dict(frame=frame,voxels=len(q),leaf_parents=len(leaf),
            pool_counts=payload['proposal_pool_counts'],native_candidates=raw_candidates,
            single_nonempty_parent_candidates=single_parent,
            supported_quality_legal_masks=len(supported),expanded_legal_masks=len(legal),
            operation_shares=payload['operation_shares'],initial_total_ratio=payload['total_ratio']))
    after=protect_snapshot(root)
    if after!=before:
        raise AssertionError('Protected source/codec files changed during audit')
    audit['protected_files_unchanged']=len(before)
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
