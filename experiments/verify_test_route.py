"""Compare restored route settings/pools against read-only source snapshots.

Usage: python -B experiments/verify_test_route.py --before SOURCE_DIR
       --historical SOURCE_DIR --out NEW_JSON
The snapshots contain train.py and models/ at their original relative paths.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(source, relative, alias):
    spec = importlib.util.spec_from_file_location(alias, str(source / relative))
    module = importlib.util.module_from_spec(spec)
    # Config resolves codec/data paths from __file__; frozen source must keep
    # its production location, just like the existing FrozenSources runner.
    module.__file__ = str(ROOT / relative)
    exec(compile((source / relative).read_text(), str(ROOT / relative), 'exec'), module.__dict__)
    return module


def parse(module, options):
    previous = sys.argv
    try:
        sys.argv = ['route_audit'] + options
        return module.parse_pugan_args(argparse.ArgumentParser(), '20261028', 'audit')
    finally:
        sys.argv = previous


def stable_pool(payload):
    return {key: value for key, value in payload.items()
            if key not in ('elapsed_sec', 'proposal_timing')}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--historical', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    from models.utils.config import args as current
    from models.utils.pointcloud import ana_den6_online as pool_module
    from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
    old = load(options.before, 'models/utils/config/args.py', 'models.utils.config._before_args')
    before_pool = load(options.before, 'models/utils/pointcloud/ana_den6_online.py',
                       'models.utils.pointcloud._before_pool')
    historical_pool = load(options.historical, 'models/utils/pointcloud/ana_den6_online.py',
                           'models.utils.pointcloud._historical_pool')
    historical_config = load(options.historical, 'models/utils/training/sparsepcgc_autonomous_policy.py',
                             'models.utils.training._historical_config')
    comparisons = []
    for dataset in ('8i', 'MVUB', 'UVG'):
        argv = ['--compression', 'SparsePCGC', '--dataname', dataset]
        baseline = parse(old, argv)
        default = parse(current, argv)
        explicit_false = parse(current, argv + ['--test_route', 'false'])
        # Every pre-existing argument keeps its original value, including loss
        # weights, Amount/Where controls, codec settings and data selection.
        for candidate in (default, explicit_false):
            differences = [key for key, value in vars(baseline).items()
                           if getattr(candidate, key) != value]
            assert not differences, differences
            assert not candidate.test_route
        experimental = parse(current, argv + ['--test_route', 'true'])
        historical = copy.deepcopy(baseline)
        historical.sparsepcgc_autonomous_search = True
        historical_config.configure_autonomous_search(historical)
        differences = [key for key, value in vars(historical).items()
                       if key != 'sparsepcgc_autonomous_search'
                       and getattr(experimental, key) != value]
        assert not differences, differences
        comparisons.append(dict(dataset=dataset, legacy_arguments=len(vars(baseline)),
                                false_equal=True, true_configuration_equal=True))
    coords = torch.cartesian_prod(torch.arange(2, 20), torch.arange(2, 20), torch.arange(2, 20))
    coords = coords[coords.sum(dim=1) % 3 == 0].T.unsqueeze(0)
    context = dict(global_voxel_coords=coords)
    identity = dict(input_sha256='3' * 64, setting_id='same_codec')
    pool_checks = []
    for route in (False, True):
        for dataset in ('8i', 'MVUB', 'UVG'):
            for step in (0, 199, 200):
                args = SimpleNamespace(compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online',
                    test_route=route, dataname=dataset, sparsepcgc_scale_m=8,
                    sparsepcgc_native_resolution=31, _global_train_step=step, _fixed_episode_steps=100,
                    heuristic_guidance_online_max_total_ratio=.0099)
                rng = torch.get_rng_state().clone()
                source = historical_pool if route else before_pool
                expected = stable_pool(source._build_fast_unordered_teacher(context, args, identity))
                actual = stable_pool(pool_module._build_fast_unordered_teacher(context, args, identity))
                assert expected == actual, (route, dataset, step)
                assert torch.equal(torch.get_rng_state(), rng)
                pool_checks.append(dict(test_route=route, dataset=dataset, step=step,
                    equal=True, torch_rng_unchanged=True,
                    counts=actual['proposal_pool_counts'],
                    sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()))
    options.out.write_text(json.dumps(dict(settings=comparisons, pools=pool_checks), indent=2))
    print('PASS: all original args, historical experiment settings, 18 pools and RNG match')


if __name__ == '__main__':
    main()
