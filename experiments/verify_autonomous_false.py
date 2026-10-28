"""Read-only False-route settings/pool/RNG regression against a source snapshot."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from experiments.verify_test_route import load, parse, stable_pool


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    from models.utils.config import args as current
    from models.utils.pointcloud import ana_den6_online as current_pool
    from models.utils.training.sparsepcgc_autonomous_policy import configure_autonomous_search
    old = load(options.before, 'models/utils/config/args.py', 'models.utils.config._false_snapshot')
    old_pool = load(options.before, 'models/utils/pointcloud/ana_den6_online.py', 'models.utils.pointcloud._false_snapshot')
    checks = []
    for dataset in ('8i', 'MVUB', 'UVG'):
        argv = ['--compression', 'SparsePCGC', '--dataname', dataset]
        baseline = parse(old, argv)
        for explicit in ([], ['--test_route', 'false']):
            actual = parse(current, argv+explicit)
            configure_autonomous_search(actual)
            diff = [k for k, v in vars(baseline).items() if getattr(actual, k) != v]
            assert not diff, diff
            checks.append(dict(dataset=dataset, explicit=bool(explicit),
                               original_arguments=len(vars(baseline)), differences=diff))
    coords = torch.cartesian_prod(torch.arange(2, 20), torch.arange(2, 20), torch.arange(2, 20))
    coords = coords[coords.sum(dim=1)%3==0].T.unsqueeze(0)
    context = dict(global_voxel_coords=coords)
    identity = dict(input_sha256='3'*64, setting_id='same_codec')
    pools = []
    for dataset in ('8i', 'MVUB', 'UVG'):
        for step in (0, 199, 200):
            args = SimpleNamespace(compress='SparsePCGC', heuristic_guidance_mode='ana_den6_online',
                test_route=False, dataname=dataset, sparsepcgc_scale_m=8,
                sparsepcgc_native_resolution=31, heuristic_guidance_online_max_total_ratio=.0099,
                heuristic_guidance_online_fast_pool_max_per_operation=8192, _global_train_step=step)
            rng = torch.get_rng_state().clone()
            expected = stable_pool(old_pool._build_fast_unordered_teacher(context, args, identity))
            actual = stable_pool(current_pool._build_fast_unordered_teacher(context, args, identity))
            assert expected == actual, (dataset, step)
            assert torch.equal(rng, torch.get_rng_state())
            pools.append(dict(dataset=dataset, step=step, equal=True, rng_equal=True))
    options.out.write_text(json.dumps(dict(settings=checks, pools=pools), indent=2))
    print('PASS: original False arguments, 9 pools and RNG')


if __name__ == '__main__':
    main()
