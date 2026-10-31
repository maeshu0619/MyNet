"""Isolated real train.py timing/parity run; never writes production artifacts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed-cache', required=True, type=Path)
    parser.add_argument('--steps', default=4, type=int)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--teacher-device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--bins', default='0.001,0.009,0.01,0.015,0.03')
    parser.add_argument('--test-route', default='true', choices=['true', 'false'])
    parser.add_argument('--source-snapshot', type=Path,
                        help='Read-only import replay for matched pre-optimization timings')
    parser.add_argument('--cpu-profile-step', type=int, default=0,
                        help='Optional cProfile of a single full forward/backward update')
    parser.add_argument('--profile-full-step', action='store_true',
                        help='Continue the profile past optimizer/logging until the next forward')
    parser.add_argument('--read-only-tensor-cache', action='store_true',
                        help='Read existing seed tensors on misses, write only private cache')
    parser.add_argument('--native-stream-audit', action='store_true',
                        help='Record real stream SHA256 and native scalar metrics for parity')
    parser.add_argument('--native-morton-stats', choices=['true', 'false'])
    parser.add_argument('--native-exact-loader', choices=['true', 'false'])
    parser.add_argument('--native-worker-trim-interval', type=int,
                        help='Private benchmark allocator lifetime; use same value in both runs')
    parser.add_argument('--episode-frames',type=int,default=4)
    parser.add_argument('--readonly-den6-cache',type=Path,
                        help='Read existing immutable den6 teachers on a private-cache miss')
    parser.add_argument('--keep-training-rng',action='store_true',help='Keep normal continuous training RNG rather than per-step diagnostic seeds')
    parser.add_argument('--lightweight', action='store_true',
                        help='Measure normal wall time without forced CUDA syncs or per-step tensor dumps')
    options = parser.parse_args()
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from experiments.sparsepcgc_direction.common import source_hashes
    if options.source_snapshot:
        from experiments.sparsepcgc_direction.common import FrozenSources
        sys.meta_path.insert(0, FrozenSources(options.source_snapshot.resolve()))
        (out/'loaded_source_snapshot.json').write_text(json.dumps(dict(
            directory=str(options.source_snapshot.resolve()),
            hashes=json.loads((options.source_snapshot/'manifest.json').read_text())), indent=2))
    initial_sources = source_hashes()
    (out/'source_sha256.json').write_text(json.dumps(initial_sources, indent=2))
    for folder in ('features', 'online', 'structure', 'tensors'):
        if folder == 'tensors' and options.read_only_tensor_cache:
            continue
        source = options.seed_cache/folder
        if source.is_dir():
            shutil.copytree(str(source), str(out/'cache'/folder))
    registry = out/'registry/SparsePCGC'
    registry.mkdir(parents=True)
    shutil.copy2('/data/maejima/data/pretrained_surrogate/SparsePCGC/20260729_001750.pth', str(registry))
    import torch
    import train as training
    from models.utils.training import train_runtime as runtime
    from models.utils.pointcloud import autonomous_proposals as proposals
    if options.readonly_den6_cache:
        import copy
        from models.utils.pointcloud import ana_den6_online
        original_teacher=ana_den6_online._load_exact_single_plan_teacher
        def readonly_teacher(settings,identity):
            teacher=original_teacher(settings,identity)
            if teacher is None:
                readonly=copy.copy(settings)
                readonly.heuristic_guidance_online_cache_dir=str(options.readonly_den6_cache)
                teacher=original_teacher(readonly,identity)
            return teacher
        ana_den6_online._load_exact_single_plan_teacher=readonly_teacher
    if options.read_only_tensor_cache:
        import copy
        from models.modules import heuristic_guidance
        original_load = heuristic_guidance._load_exact_guidance_disk
        def load_with_readonly_seed(settings, key):
            result = original_load(settings, key)
            if result is None:
                readonly = copy.copy(settings)
                readonly.heuristic_guidance_tensor_disk_cache_dir = str(options.seed_cache/'tensors')
                result = original_load(readonly, key)
            return result
        heuristic_guidance._load_exact_guidance_disk = load_with_readonly_seed
    state = dict(step=0, times=defaultdict(float), phase='other_backward')
    original_parse = training.parse_pugan_args

    def parse(*args, **kwargs):
        settings = original_parse(*args, **kwargs)
        settings.seed = int(options.seed)
        return settings

    training.parse_pugan_args = parse

    def sync():
        if not options.lightweight and torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed(label, function, phase=None):
        def call(*args, **kwargs):
            previous = state['phase']
            if phase is not None:
                state['phase'] = phase
            sync()
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                sync()
                state['times'][label] += time.perf_counter() - start
                state['phase'] = previous
        return call

    proposals.build_autonomous_pool = timed('pool_generation', proposals.build_autonomous_pool)
    original_network = training.Network

    def network(*args, **kwargs):
        model = original_network(*args, **kwargs)
        state['model'] = model
        model.actuator._build_exact_den6_residual_plan = timed(
            'plan_selection', model.actuator._build_exact_den6_residual_plan)
        model.actuator.forward = timed('actuator_forward', model.actuator.forward)
        original_forward = model.forward

        def forward(*args, **kwargs):
            if not model.training:
                return original_forward(*args, **kwargs)
            if options.profile_full_step and state.get('cpu_profiler') is not None:
                previous_profiler = state.pop('cpu_profiler')
                previous_profiler.disable()
                previous_profiler.dump_stats(str(out/'cpu_profile.pstats'))
            state['step'] += 1
            state['times'].clear()
            state['native_calls'] = []
            if not options.keep_training_rng:
                runtime.set_seed(options.seed * 1000003 + state['step'] - 1, deterministic=False)
            torch.cuda.reset_peak_memory_stats()
            if state['step'] == options.cpu_profile_step:
                import cProfile
                state['cpu_profiler'] = cProfile.Profile()
                state['cpu_profiler'].enable()
            return timed('network_forward', original_forward, 'forward')(*args, **kwargs)

        model.forward = forward
        return model

    training.Network = network
    original_loss = training.Loss

    def loss(*args, **kwargs):
        obj = original_loss(*args, **kwargs)
        state['loss'] = obj
        obj.get_compression_loss = timed('compression_loss', obj.get_compression_loss, 'surrogate_backward')
        obj.get_geometry_loss = timed('geometry_loss', obj.get_geometry_loss)
        obj._encode_actual_batch = timed('actual_encode', obj._encode_actual_batch)
        if options.native_stream_audit:
            original_get = obj._get_actual_encoder
            def get_encoder(settings):
                encoder = original_get(settings)
                if not getattr(encoder, '_speed_stream_audit', False):
                    encoder._speed_stream_audit = True
                    original_encode = encoder.encode_bits
                    original_send = encoder._send_worker_request_with_oom_retry
                    def send(*positional, **named):
                        result = original_send(*positional, **named)
                        state['latest_native_response'] = {k: v for k, v in result[0].get('result', {}).items()
                            if k in ('point_count', 'node_count', 'single_child_count', 'stats_time',
                                     'wall_time_seconds', 'enc_time', 'all_enc_time', 'file_size')}
                        return result
                    encoder._send_worker_request_with_oom_retry = send
                    def encode(*positional, **named):
                        import hashlib
                        state['latest_native_response'] = {}
                        result = original_encode(*positional, **named)
                        streams = []
                        workspace = getattr(encoder, '_workspace_dir', None)
                        if workspace and not result.get('sparsepcgc_actual_result_cache_hit', False):
                            folder = Path(workspace)/'encoded'
                            for path in sorted(folder.rglob('*.bin')):
                                content = path.read_bytes()
                                streams.append(dict(file=str(path.relative_to(folder)), bits=8*len(content),
                                                    sha256=hashlib.sha256(content).hexdigest()))
                        state.setdefault('native_calls', []).append(dict(
                            metrics={k: result.get(k) for k in ('bit', 'point_count', 'node', 'single', 'sparsepcgc_point_sha256')},
                            worker=dict(state['latest_native_response']), streams=streams))
                        return result
                    encoder.encode_bits = encode
                return encoder
            obj._get_actual_encoder = get_encoder
        return obj

    training.Loss = loss
    original_capture = training._capture_den6_policy_only_gradients
    training._capture_den6_policy_only_gradients = timed('policy_capture', original_capture, 'capture_backward')
    original_backward = torch.Tensor.backward

    def backward(tensor, *args, **kwargs):
        return timed(state['phase'], original_backward)(tensor, *args, **kwargs)

    torch.Tensor.backward = backward
    original_build = training.build_optimizer_and_scheduler

    def build(model, args, writer):
        (out/'effective_args.json').write_text(json.dumps(vars(args), indent=2, default=str))
        if not options.lightweight:
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, str(out/'initial.pth'))
        return original_build(model, args, writer)

    training.build_optimizer_and_scheduler = build
    original_step = torch.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        model = state.get('model')
        ids = {id(p) for p in model.actuator.algorithmic_amount_selector_head.parameters()} if model is not None else set()
        is_main = any(id(p) in ids for group in optimizer.param_groups for p in group['params'])
        if not is_main:
            return original_step(optimizer, *args, **kwargs)
        gradients = ({n: p.grad.detach().cpu().clone() if p.grad is not None else None
                      for n, p in model.named_parameters() if p.requires_grad}
                     if not options.lightweight else {})
        result = timed('optimizer', original_step)(optimizer, *args, **kwargs)
        profiler = state.pop('cpu_profiler', None) if not options.profile_full_step else None
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(out/'cpu_profile.pstats'))
        debug = model.last_actuator_voxel_state['ana_den6_exact_residual_plan_debug']
        row = dict(step=state['step'], times=dict(state['times']),
                   geometry_nn=getattr(state['loss'], 'last_autonomous_geometry_nn', []),
                   peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                   policy=dict(model.last_discrete_policy_debug),
                   plan={k: debug.get(k) for k in ('selected_counts', 'selected_amount_ratios',
                        'final_edited_voxel_hash', 'joint_amount_bins_by_operation', 'candidate_pool_sizes')})
        frozen_cache = getattr(model, '_autonomous_frozen_feature_ram', None)
        if frozen_cache is not None:
            row['frozen_ram_cache'] = frozen_cache.stats()
        pool_cache = getattr(proposals, '_POOL_RAM', None)
        if pool_cache is not None:
            row['pool_ram_cache'] = pool_cache.stats()
        if options.native_stream_audit:
            row['native_calls'] = state.get('native_calls', [])
        with (out/'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row, default=str)+'\n')
        if not options.lightweight:
            torch.save(dict(gradients=gradients,
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}),
                str(out/('step_%03d.pth' % state['step'])))
        return result

    torch.optim.Adam.step = step
    sys.argv = ['train.py', '--test_route', options.test_route,
        '--test_route_amount_bins', options.bins, '--compression', 'SparsePCGC',
        '--seed', str(options.seed), '--more_training', 'false', '--mail_notify', 'false',
        '--train_fixed_episode_frames', str(options.episode_frames), '--max_train_steps', str(options.steps),
        '--train_full_cloud_val_frames', '0', '--sparsepcgc_device', options.teacher_device,
        '--heuristic_guidance_online_grad_audit', 'false',
        '--heuristic_guidance_online_source_grad_probe', 'false',
        '--log_root', str(out/'log'), '--out_path', str(out/'checkpoints'),
        '--surrogate_pretrained_root', str(out/'registry'), '--debug_timing', 'true',
        '--profile_interval', '1']
    for value, flag in ((options.native_morton_stats, 'test_route_native_morton_stats'),
                        (options.native_exact_loader, 'test_route_native_exact_loader'),
                        (options.native_worker_trim_interval, 'sparsepcgc_worker_cpu_trim_interval')):
        if value is not None:
            sys.argv += ['--'+flag, str(value)]
    for flag, folder in (
        ('point_transformer_feature_disk_cache_dir', 'features'),
        ('sparsepcgc_actual_gt_disk_cache_dir', 'actual_gt'), ('exact_teacher_cache_root', 'exact'),
        ('heuristic_guidance_online_cache_dir', 'online'), ('heuristic_guidance_tensor_disk_cache_dir', 'tensors'),
        ('structure_fixed_disk_cache_dir', 'structure'), ('surrogate_pretrain_cache_dir', 'surrogate'),
        ('sparsepcgc_tmp_dir', 'tmp')):
        sys.argv += ['--'+flag, str(out/'cache'/folder)]
    (out/'invocation.json').write_text(json.dumps(sys.argv, indent=2))
    try:
        training.main()
        (out/'completed.json').write_text(json.dumps(dict(updates=state['step'])))
    finally:
        current = source_hashes()
        changed = [p for p, h in initial_sources.items() if current.get(p) != h]
        (out/'integrity.json').write_text(json.dumps(dict(source_changes=changed)))
        assert not changed


if __name__ == '__main__':
    main()
