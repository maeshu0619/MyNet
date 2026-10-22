"""Six-point, physical-stream D1 pilot for an existing frozen MINE output.

D2 is deliberately unavailable without supplied reference normals. This is not
a new model inference, a full test-set evaluation, or the training logical rate.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from utils.bdrate.codecs import default_points,encode_decode
from utils.bdrate.io import read_vertices,save_json,sha256,write_geometry
from utils.bdrate.metrics import evaluate
from utils.bdrate.report import summarize,write_report,plot_curves
from experiments.sparsepcgc_direction.common import geometry


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    options=parser.parse_args();out=options.root/'rd_frame0';out.mkdir(exist_ok=False)
    manifest=json.loads((options.root/'cases.json').read_text())
    selected=[c for c in manifest['cases'] if c['id'].startswith('frame0_')]
    gt=Path('/data/maejima/data/ground/UVG/BlueBackpack')/(selected[0]['frame']+'.ply')
    gt_xyz,gt_vertices=read_vertices(gt)
    baseline=np.load(selected[0]['npz'])['points']
    if not np.array_equal(np.unique(gt_xyz,axis=0),np.unique(baseline,axis=0)):
        raise ValueError('Saved baseline is not the Original GT geometry')
    has_normals=all(name in gt_vertices.dtype.names for name in ['nx','ny','nz'])
    if has_normals:raise ValueError('Pilot expects known absent normals; select explicit normals protocol instead')
    codec=ROOT.parent/'compress/octree/SparsePCGC'
    settings=dict(codec_device='cpu',threads=2,timeout=1800,
                  sparse_python='/home/maejima/miniconda3/envs/sparsepcgc/bin/python',
                  sp_checkpoint=str(codec/'ckpts/dense/epoch_last.pth'),
                  sp_sr_checkpoint=str(codec/'ckpts/dense_1stage/epoch_last.pth'),
                  sp_ae_checkpoint=str(codec/'ckpts/dense_slne/epoch_last.pth'))
    rows=[];cloud=selected[0]['frame'];points=default_points('SparsePCGC')
    for case in selected:
        source=out/(case['variant']+'.ply');write_geometry(source,np.load(case['npz'])['points'])
        for index,point in enumerate(points):
            folder=out/('%s_%02d'%(case['variant'],index));folder.mkdir()
            row=dict(cloud=cloud,series='Baseline' if case['variant']=='baseline' else 'MINE',
                     rate_index=index,point=point,original_points=len(gt_xyz),
                     mine_points=case['points'],status='error')
            try:
                reused=options.root/'native_v3'/case['id']
                if point==dict(scale_ae=0,scale_sr=1) and (reused/'trace.json').is_file():
                    trace=json.loads((reused/'trace.json').read_text())
                    bits,streams=trace['stream_measurement']
                    decoded=reused/'decoded.ply';xyz,_=read_vertices(decoded)
                    encoded=dict(bits=bits,decoded_path=str(decoded),decoded_points=len(xyz),
                                 streams=streams,codec_metadata=trace['codec'],reused_native_trace=True)
                else:
                    encoded=encode_decode('SparsePCGC',source,folder,point,settings)
                row.update(encoded,bpp=encoded['bits']/len(gt_xyz))
                row.update(evaluate(gt,encoded['decoded_path'],folder,
                    str(codec/'extension/pc_error_d'),resolution=511,require_d2=False))
                decoded_xyz,_=read_vertices(encoded['decoded_path'])
                row.update(geometry(gt_xyz,decoded_xyz),status='ok')
            except Exception as error:
                row['error']=type(error).__name__+': '+str(error)
            save_json(folder/'measurement.json',row);rows.append(row)
            print('[RD]',row['series'],index,row['status'],row.get('bits'),row.get('d1_psnr'),flush=True)
    summaries,diagnostics=summarize(rows,[cloud],len(points),list(range(6)))
    metadata=dict(scope='ONE_FRAME_FROZEN_NETWORK_OUTPUT_REPLAY_NOT_FULL_TESTSET',
                  checkpoint=manifest['source_options']['checkpoint'],
                  checkpoint_sha256=sha256(manifest['source_options']['checkpoint']),
                  new_network_inferences=0,source_manifest=str(options.root/'cases.json'),
                  GT=str(gt),GT_sha256=sha256(gt),resolution=511,
                  D2_status='not measured: reference GT has no nx/ny/nz; normals not invented',
                  codec_settings=settings,rate_points=points,
                  evaluation_status=dict(successful_rd_points=sum(r['status']=='ok' for r in rows),
                                         requested_rd_points=12))
    write_report(out,rows,summaries,diagnostics,metadata);plot_curves(out,rows,[cloud])
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
