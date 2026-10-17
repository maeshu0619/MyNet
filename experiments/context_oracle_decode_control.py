#!/usr/bin/env python
"""Execute only official OA decoder functions on a small CPU control.

AST extraction avoids decoder.py's GPU/global encoder imports. The function
bodies themselves are unmodified. Full-frame serial decoding is not attempted.
"""
import argparse
import ast
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from context_oracle_core import OAActual,unique


class SilentBar:
    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def update(self,*args):
        pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(2);folder=Path(a.out);folder.mkdir(parents=True,exist_ok=False)
    oa=OAActual('cpu');q=unique(np.random.RandomState(51).randint(0,8,(128,3)))
    before,seq,prob=oa.encode(q,detail=True)
    import numpyAc
    from Octree import dec2bin,DeOctree
    symbols=(seq[:,-1,0]-1).astype(np.int16)
    filename=str(folder/'small_control.bin')
    numpyAc.arithmeticCoding().encode(prob,symbols,filename)
    path=Path(__file__).resolve().parents[2]/'compress/octree/OctAttention/decoder.py'
    tree=ast.parse(path.read_text())
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('decodeOct','decodeNode')]
    namespace=dict(np=np,torch=torch,time=time,deque=deque,numpyAc=numpyAc,
        levelNumK=4,batch_size=1,bpttRepeatTime=1,device=torch.device('cpu'),
        generate_square_subsequent_mask=oa.teacher._generate_square_subsequent_mask,
        dec2bin=dec2bin,tqdm=SilentBar)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)
    started=time.perf_counter()
    try:
        decoded,elapsed=namespace['decodeOct'](filename,seq[:,-1,0],oa.teacher._model,min(1024,len(seq)-1))
        exact_symbols=np.array_equal(np.array(decoded),seq[:,-1,0])
        exact_points=np.array_equal(unique(DeOctree(np.array(decoded))),q-q.min(0))
        result=dict(official_function_body_unmodified=True,nodes=len(seq),voxels=len(q),
            actual_bits=before['bits'],symbol_roundtrip=exact_symbols,geometry_roundtrip=exact_points,
            autonomous_probabilities=True,elapsed_sec=elapsed)
    except Exception as error:
        result=dict(official_function_body_unmodified=True,nodes=len(seq),voxels=len(q),
            autonomous_probabilities=True,symbol_roundtrip=False,geometry_roundtrip=False,
            error_type=type(error).__name__,error=str(error),elapsed_sec=time.perf_counter()-started)
    (folder/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
