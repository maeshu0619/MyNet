"""Private actual-codec timing/parity probe; never edits codec sources.

One persistent model serves every case. GPU contention is retained and
reported; alternating orders reduce but do not eliminate timing confounds.
"""
import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'models/utils/loss'))


def coordinates_hash(tensor):
    rows = tensor.C[:, 1:].detach().cpu().numpy().astype(np.int64)
    rows = rows[np.lexsort((rows[:,2],rows[:,1],rows[:,0]))]
    return hashlib.sha256(rows.tobytes()).hexdigest()


def require_host_capacity(minimum_mib=10000):
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))/1024.
    if available < minimum_mib:
        raise RuntimeError('Private probe stopped: host available %.1f MiB < %.1f MiB' %
                           (available, minimum_mib))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--modes', default='native,isin,sort,both,nodebug')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--include-edited', action='store_true')
    parser.add_argument('--unique-reference',type=Path)
    parser.add_argument('--warmup',action='store_true')
    options=parser.parse_args(); options.out.mkdir(parents=True,exist_ok=False)
    import torch
    import sparsepcgc_teacher_worker as worker
    from experiments.native_symbol_preparation import install_symbol_preparation
    from experiments.native_codec_repeat_decode import mixed_edit, write_input
    require_host_capacity(16000)
    if torch.cuda.mem_get_info()[0] < 10*1024**3:
        raise RuntimeError('Insufficient shared capacity for private native probe')
    codec=ROOT.parent/'compress/octree/SparsePCGC'
    sys.argv=['probe','--sparsepcgc-root',str(codec),'--mode','dense_lossy','--device','cuda',
        '--scale-m','8','--scale-ae','0','--scale-sr','2',
        '--exact-unique-loader','--exact-morton-stats','--exact-logical-rate']
    for flag,folder in [('ckptdir','dense'),('ckptdir-sr','dense_1stage'),('ckptdir-ae','dense_slne'),
                        ('ckptdir-low','sparse_low'),('ckptdir-high','sparse_high'),('ckptdir-offset','sparse_offset')]:
        sys.argv += ['--'+flag,str(codec/'ckpts'/folder/'epoch_last.pth')]
    loads=[];original_load=torch.load
    def load(*args,**kwargs):
        loads.append(str(args[0]) if args else '<keyword>')
        return original_load(*args,**kwargs)
    torch.load=load
    start=time.perf_counter();encoder=worker._build_encoder(worker._parse_args());build_seconds=time.perf_counter()-start
    original_one=encoder.encode_one
    import autonomous_native_loader as loader
    fast_unique=loader.lexicographic_unique_rows
    reference_unique=fast_unique
    if options.unique_reference:
        import importlib.util
        spec=importlib.util.spec_from_file_location('_native_unique_reference',str(options.unique_reference))
        reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
        reference_unique=reference_module.lexicographic_unique_rows
    basic=encoder.coder.basic_coder;upsampler=basic.model.upsampler;original_encode=upsampler.encode
    native=sys.modules[type(encoder).__module__]
    loader_namespace=basic.load_data.__func__.__globals__['load_sparse_tensor'].__globals__
    fast_reader=loader_namespace['read_coords']
    original_reader=getattr(fast_reader,'_mynet_native_reader',fast_reader)
    source, _=native._read_ply_points(options.input)
    inputs=[('gt',np.asarray(source,dtype=np.float32))]
    if options.include_edited:
        inputs.append(('edited',mixed_edit(source)[0]))
    events={};capture={};stage_times={}
    def timed(obj,name,label):
        original=getattr(obj,name)
        def wrapped(*a,**kw):
            events[label]=events.get(label,0)+1;start=time.perf_counter()
            try:return original(*a,**kw)
            finally:stage_times[label]=stage_times.get(label,0.)+time.perf_counter()-start
        setattr(obj,name,wrapped)
    timed(basic,'load_data','load_data')
    timed(encoder.coder,'downscale','downscale')
    timed(basic,'encode','basic_encode')
    timed(basic,'pack_bitstream','pack')
    downscale=encoder.coder.downscale
    def observed_downscale(*a,**kw):
        value=downscale(*a,**kw)
        capture.update(ae=value[1],counts=list(value[2]),low_hash=coordinates_hash(value[0]))
        return value
    encoder.coder.downscale=observed_downscale
    modes=options.modes.split(','); records=[]
    for input_name,points in inputs:
        input_file=options.out/(input_name+'.ply');write_input(input_file,points)
        if options.warmup:
            loader.lexicographic_unique_rows=reference_unique
            encoder.encode_one=original_one;upsampler.encode=original_encode
            encoder.encode_one(input_file,options.out/(input_name+'_warmup'))
            encoder._full_cleanup_runtime_cache()
        for repeat in range(options.repeats):
            order=modes if repeat%2==0 else list(reversed(modes))
            for mode in order:
                require_host_capacity()
                if mode not in ('native','reader','isin','sort','both','nodebug','combined','unique','exact','exact_no_debug','gpu_isin','gpu_exact'):
                    raise ValueError(mode)
                loader_namespace['read_coords']=fast_reader if mode=='reader' else original_reader
                loader.lexicographic_unique_rows=fast_unique if mode in ('unique','exact','exact_no_debug','gpu_exact') else reference_unique
                encoder.encode_one=original_one;upsampler.encode=original_encode
                if mode not in ('native','unique','reader'):
                    install_symbol_preparation(encoder,membership='gpu' if mode in ('gpu_isin','gpu_exact') else mode in ('isin','both','combined'),
                        sorting=mode in ('sort','both','combined','exact','exact_no_debug','gpu_exact'),verify=options.verify,
                        debug_decode=mode not in ('nodebug','combined','exact_no_debug'))
                folder=options.out/('%s_%d_%s'%(input_name,repeat,mode))
                stage_times.clear();before=len(loads);profiler=cProfile.Profile();start=time.perf_counter()
                if options.profile:profiler.enable()
                try:
                    result=dict(encoder.encode_one(input_file,folder))
                    encoder._full_cleanup_runtime_cache()
                finally:
                    if options.profile:
                        profiler.disable();profiler.dump_stats(str(options.out/('%s_%d_%s.pstats'%(input_name,repeat,mode))))
                elapsed=time.perf_counter()-start
                stream_path=next(folder.glob('*.bin'));stream=stream_path.read_bytes()
                row=dict(input=input_name,repeat=repeat,mode=mode,verify=options.verify,
                    encode_cleanup_seconds=elapsed,stages=dict(stage_times),new_checkpoint_loads=len(loads)-before,
                    logical_bits=result['file_size'],physical_bits=len(stream)*8,
                    stream_sha256=hashlib.sha256(stream).hexdigest(),
                    input_hash=hashlib.sha256(points.tobytes()).hexdigest(),
                    native_metrics={k:result[k] for k in ('point_count','node_count','single_child_count')},
                    symbol_counts=result.get('native_symbol_preparation_counts'),
                    peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
                    stats_time=result.get('stats_time'),low_hash=capture['low_hash'])
                if options.decode:
                    require_host_capacity()
                    streams=basic.unpack_bitstream(str(stream_path));decoded_low=basic.decode(streams)
                    row['decoded_low_hash']=coordinates_hash(decoded_low)
                    assert row['decoded_low_hash']==row['low_hash']
                    decoded=encoder.coder.upscale(decoded_low,capture['ae'],list(capture['counts']))
                    row['decoded_hash']=coordinates_hash(decoded);row['decoded_points']=len(decoded)
                    del decoded,decoded_low
                    encoder._full_cleanup_runtime_cache();gc.collect()
                records.append(row)
                with (options.out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    (options.out/'summary.json').write_text(json.dumps(dict(build_seconds=build_seconds,
        checkpoint_loads=loads,events=events,records=records),indent=2))


if __name__=='__main__':main()
