"""Capture actual structure features during the existing fixed-validation path."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-dir',type=Path,required=True)
    opts,extra=parser.parse_known_args()
    opts.capture_dir.mkdir(parents=True,exist_ok=False)
    from models.modules.octree_structure import OctreeStructureAnalysis
    original=OctreeStructureAnalysis.forward
    records=[]
    def forward(owner,*args,**kwargs):
        result=original(owner,*args,**kwargs)
        frame=str(getattr(owner.args,'_current_input_file',''))
        path=opts.capture_dir/('%03d.npz'%len(records))
        coords=result.get('global_voxel_coords')
        if coords is None:raise ValueError('Canonical feature coordinates missing')
        np.savez_compressed(str(path),features=result['features'].detach().cpu().numpy(),
                            causes=result['cause_targets'].detach().cpu().numpy(),
                            occupancy_nll_proxy=result['occupancy_nll_proxy'].detach().cpu().numpy(),
                            coords=coords.detach().cpu().numpy(),
                            xyz=args[0].detach().cpu().numpy())
        record=dict(frame=frame,file=path.name,points=int(result['features'].shape[-1]),
                    octree_input_mode=result['octree_input_mode'],training=owner.training)
        records.append(record)
        (opts.capture_dir/'index.json').write_text(json.dumps(records,indent=2))
        print('[FEATURE]',record,flush=True)
        return result
    OctreeStructureAnalysis.forward=forward
    from experiments.sparsepcgc_direction.run import main as run
    sys.argv=[sys.argv[0]]+extra
    run()


if __name__=='__main__':main()
