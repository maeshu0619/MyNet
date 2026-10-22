"""Instrument native SparsePCGC in this subprocess only; reuse its real RD adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from utils.bdrate.io import save_json,write_geometry,measure_streams
from utils.bdrate.learned_worker import sparsepcgc


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--name',default='native_v2')
    options=parser.parse_args();out=options.root/options.name;out.mkdir(exist_ok=False)
    codec=ROOT.parent/'compress/octree/SparsePCGC'
    sys.path.insert(0,str(codec));sys.path.insert(0,str(codec/'test'))
    import torch
    import torchac
    from torchac.torchac import _convert_to_int_and_normalize
    # PCCModel imports decoder as a top-level module inside its constructor.
    # Patching models.decoder would instrument a different Python class.
    sys.path.insert(0,str(codec/'models'))
    from decoder import UpSampler8Stage,BinaryArithmeticCoding
    from data_utils.sparse_tensor import sort_sparse_tensor
    torch.set_num_threads(2)
    state=dict(record=False,level=-1,stages=[],pending=[],folder=None)
    original_encode=UpSampler8Stage.encode
    original_forward=UpSampler8Stage.forward
    original_bac=BinaryArithmeticCoding.encode

    def forward(owner,x_low,x_high=None,*args,**kwargs):
        result=original_forward(owner,x_low,x_high,*args,**kwargs)
        if state['record']:
            state['pending']=[sort_sparse_tensor(x).C[:,1:].cpu().numpy().copy()
                              for x in result['out_cls_list']]
        return result

    def encode(owner,x_low,x_high,*args,**kwargs):
        state['level']+=1;state['stage']=0;state['record']=True
        try:return original_encode(owner,x_low,x_high,*args,**kwargs)
        finally:state['record']=False

    def bac(owner,prob,occupancy,*args,**kwargs):
        stream=original_bac(owner,prob,occupancy,*args,**kwargs)
        if not state['record']:return stream
        stage=state['stage'];state['stage']+=1
        coords=state['pending'][stage]
        p=prob.reshape(-1).double().numpy();symbols=occupancy.reshape(-1).numpy().astype(np.int8)
        cdf=owner._get_cdf(prob)
        # Call exactly the integer conversion used by encode_float_cdf.
        integers=_convert_to_int_and_normalize(cdf,True).to(torch.int64).numpy()%65536
        integers=integers.reshape(-1,3);integers[:,-1]=65536
        widths=np.diff(integers,axis=1)
        actual_probability=widths[np.arange(len(symbols)),symbols]/65536.
        if np.any(actual_probability<=0):raise ValueError('Invalid native integer CDF')
        nll=-np.log2(actual_probability)
        true_float=np.where(symbols>0,p,1-p)
        with np.errstate(divide='ignore'):float_nll=-np.log2(true_float)
        file='level_%02d_stage_%02d.npz'%(state['level'],stage)
        np.savez_compressed(str(state['folder']/file),coords=coords,occupancy=symbols,
                            probability=p,cdf_probability=actual_probability,
                            nll=nll,float_nll=float_nll)
        record=dict(level=state['level'],stage=stage,candidates=len(symbols),
                    occupied=int(symbols.sum()),integer_cdf_NLL=float(nll.sum()),
                    float_NLL=float(float_nll.sum()) if np.isfinite(float_nll).all() else None,
                    float_NLL_infinite_count=int((~np.isfinite(float_nll)).sum()),
                    arithmetic_bits=8*len(stream),array_file=file)
        state['stages'].append(record)
        return stream
    UpSampler8Stage.forward=forward;UpSampler8Stage.encode=encode;BinaryArithmeticCoding.encode=bac
    cases=json.loads((options.root/'cases.json').read_text())['cases']
    settings=dict(codec_device='cpu',sp_checkpoint=str(codec/'ckpts/dense/epoch_last.pth'),
                  sp_sr_checkpoint=str(codec/'ckpts/dense_1stage/epoch_last.pth'),
                  sp_ae_checkpoint=str(codec/'ckpts/dense_slne/epoch_last.pth'))
    for case in cases:
        folder=out/case['id'];folder.mkdir();points=np.load(case['npz'])['points']
        source=folder/'source.ply';write_geometry(source,points)
        state.update(record=False,level=-1,stages=[],pending=[],folder=folder)
        result=sparsepcgc(dict(settings=settings,folder=str(folder),source=str(source),
                              point=dict(scale_ae=0,scale_sr=1)))
        if not state['stages']:
            raise RuntimeError('Native Stage instrumentation did not observe any coder calls')
        streams=[Path(p) for p in result['streams']]
        counts=measure_streams(streams)
        record=dict(case=case,stages=state['stages'],codec=result,stream_measurement=counts,
                    integer_cdf_NLL=sum(r['integer_cdf_NLL'] for r in state['stages']),
                    arithmetic_payload_bits=sum(r['arithmetic_bits'] for r in state['stages']))
        save_json(folder/'trace.json',record)
        print('[TRACE]',case['id'],json.dumps({k:record[k] for k in ['integer_cdf_NLL','arithmetic_payload_bits']}),flush=True)
    protected=json.loads((options.root/'protected_sources.json').read_text())
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    save_json(out/'integrity.json',dict(protected_source_changes=changed))
    if changed:raise RuntimeError('Protected source changed externally during trace')


if __name__=='__main__':main()
