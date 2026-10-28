"""Compare isolated timing runs without rounding the underlying measurements."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch


def rows(directory):
    return [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]


def same_tensor(x, y):
    return x.shape == y.shape and torch.allclose(x, y, atol=0., rtol=0., equal_nan=True)


def summarize(directory, stdout=None):
    data = rows(directory)
    stdout = stdout or directory.parent/(directory.name+'_stdout.txt')
    wall = [float(v) for v in re.findall(r'Epi\d+/Epo\d+/Step\d+:([\d.]+)s',
        stdout.read_text())] if stdout.exists() else []
    return dict(steps=len(data), times_by_step=[row['times'] for row in data],
                wall_seconds=wall,
                wall_mean=float(np.mean(wall)) if wall else None,
                warmed_wall_mean=float(np.mean(wall[4:])) if len(wall)>4 else None,
                peak_mib=max(row['peak_mib'] for row in data),
                mean_seconds={key: float(np.mean([row['times'].get(key, 0.) for row in data]))
                              for key in data[0]['times']},
                warmed_mean_seconds={key: float(np.mean([row['times'].get(key, 0.) for row in data[4:]]))
                                     for key in data[0]['times']} if len(data)>4 else None)


def compare(before, after):
    old, new = rows(before), rows(after)
    result = []
    for a, b in zip(old, new):
        i = a['step']
        previous = torch.load(str(before/('step_%03d.pth'%i)), map_location='cpu')
        current = torch.load(str(after/('step_%03d.pth'%i)), map_location='cpu')
        differences = {}
        for group in ('model', 'gradients'):
            maximum, norm, reference_norm, equal, total, missing = 0., 0., 0., 0, 0, []
            for name, x in previous[group].items():
                y = current[group].get(name)
                if x is None or y is None:
                    if (x is None) != (y is None):
                        missing.append(name)
                    continue
                d = (x.double()-y.double()).reshape(-1)
                d = d[torch.isfinite(d)]
                maximum = max(maximum, float(d.abs().max()) if d.numel() else 0.)
                norm += float(d.square().sum())
                reference_norm += float(x[torch.isfinite(x)].double().square().sum())
                equal += int(same_tensor(x, y))
                total += 1
            differences[group] = dict(max_absolute=maximum, difference_l2=norm**.5,
                reference_l2=reference_norm**.5, relative_l2=(norm/reference_norm)**.5 if reference_norm else None,
                equal_tensors=equal, tensors=total, missing_gradient_mismatches=missing)
        result.append(dict(step=i, plan_equal=a['plan']==b['plan'],
            actual_equal=a['policy']['current_compression_value']==b['policy']['current_compression_value'],
            geometry_equal=a['policy']['current_geometry_value']==b['policy']['current_geometry_value'],
            actual_percent=a['policy']['current_compression_value'],
            geometry=a['policy']['current_geometry_value'], differences=differences))
    return result


def update_metrics(directory):
    roles = {'Add Where': ('actuator.add_head.', 'actuator.add_voxel_head.'),
        'Prune Where': ('actuator.drop_head.',),
        'Adjust Where': ('actuator.move_voxel_head.', 'actuator.subtree_move_source_head.'),
        'Add Amount': ('actuator.add_amount_head.',), 'Prune Amount': ('actuator.drop_amount_head.',),
        'Adjust Amount': ('actuator.move_amount_head.',),
        'Coarse Amount': ('actuator.algorithmic_amount_selector_head.',),
        'Actor': ('actuator.den6_candidate_actor.',), 'Proposal': ('actuator.den6_candidate_proposal.',),
        'Feature Adapter': ('point_transformer_feature_adapter.',)}
    previous = torch.load(str(directory/'initial.pth'), map_location='cpu')
    result = []
    for row in rows(directory):
        current = torch.load(str(directory/('step_%03d.pth'%row['step'])), map_location='cpu')
        modules = {}
        for label, prefixes in roles.items():
            names = [k for k, v in current['gradients'].items() if k.startswith(prefixes) and v is not None]
            size = sum(current['gradients'][k].numel() for k in names)
            g = sum(float(current['gradients'][k].double().square().sum()) for k in names)**.5
            delta = sum(float((current['model'][k].double()-previous[k].double()).square().sum()) for k in names)**.5
            theta = sum(float(previous[k].double().square().sum()) for k in names)**.5
            modules[label] = dict(postclip_l2=g, postclip_rms=g/size**.5 if size else None,
                                  update_l2=delta, relative_update=delta/theta if theta else None)
        result.append(dict(step=row['step'], plan=row['plan'], policy=row['policy'], modules=modules,
            finite_gradients=all(torch.isfinite(v).all().item() for v in current['gradients'].values() if v is not None)))
        previous = current['model']
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--before-stdout', type=Path)
    parser.add_argument('--after-stdout', type=Path)
    parser.add_argument('--expanded', type=Path)
    parser.add_argument('--expanded-stdout', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    a, b = [torch.load(str(directory/'initial.pth'), map_location='cpu')
            for directory in (args.before, args.after)]
    initial_equal = a.keys()==b.keys() and all(same_tensor(a[k], b[k]) for k in a)
    output = dict(before=summarize(args.before, args.before_stdout), after=summarize(args.after, args.after_stdout),
                  initial_weights_equal=initial_equal, comparisons=compare(args.before, args.after))
    if args.expanded:
        output['expanded'] = dict(timing=summarize(args.expanded, args.expanded_stdout),
                                  updates=update_metrics(args.expanded))
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(dict(initial_weights_equal=initial_equal,
        all_plans_equal=all(row['plan_equal'] for row in output['comparisons']),
        all_actual_equal=all(row['actual_equal'] for row in output['comparisons']),
        all_geometry_equal=all(row['geometry_equal'] for row in output['comparisons']))))


if __name__ == '__main__':
    main()
