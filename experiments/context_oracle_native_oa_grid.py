#!/usr/bin/env python
"""Native OA raw-min-first grid control, reusing the independent study.

Translating raw coordinates by their minimum turns the common-grid study into
exactly OA's native qs=2 grid. Geometry distances are translation invariant.
No original file or training argument is changed.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import context_oracle_study as study
from octree_joint_mask_research import proposals
from context_oracle_core import OAActual,load_ply,unique


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    raw=load_ply(a.frame,return_color=False).astype(np.float64)
    translated=raw-raw.min(0)
    q=unique(np.rint(translated/2.));origin=np.zeros(3)
    oa=OAActual('cpu');baseline,seq,prob=oa.encode(q,detail=True,roundtrip=True)
    from models.utils.loss.actual_encoder import _OctAttentionActualEncoder
    teacher=_OctAttentionActualEncoder(SimpleNamespace(qs=2.,octattention_actualcode=True,
        octattention_teacher_device='cpu',bptt=1024))
    native=teacher.encode_bits(torch.from_numpy(raw.T.copy()).float())
    if baseline['bits']!=native['bit']:
        raise AssertionError('Native raw OA teacher / integer canonical grid parity failed')
    candidate,audit=proposals(q,translated,origin,seq,prob,lambda x:print(json.dumps(x),flush=True))
    cache=folder/'candidates';cache.mkdir()
    np.savez_compressed(cache/(Path(a.frame).stem+'_proposals.npz'),candidates=candidate,seq=seq,origin=origin)
    (folder/'native_grid_protocol.json').write_text(json.dumps(dict(frame=a.frame,
        raw_min=raw.min(0).tolist(),native_grid='round((raw-raw_min)/2)',baseline=baseline,
        native_teacher=native,exact_bits_parity=True,proposal_audit=audit),indent=2))
    original_loader=study.load_ply
    def translated_loader(path,return_color=False):
        points=original_loader(path,return_color=return_color)
        return points-points.min(0)
    study.load_ply=translated_loader  # only this independent process's experiment binding
    sys.argv=[sys.argv[0],'--frames',a.frame,'--out',str(folder/'study'),
              '--candidates',str(cache),'--budgets','.05']
    study.main()


if __name__=='__main__':
    main()
