"""Compare saved real-training runs without launching training or a codec.

Inputs must be trusted private outputs of autonomous_speed_probe.py (torch.load
on this repository's PyTorch version uses pickle). Exact scalar/plan
agreement and floating-point parameter differences are reported separately.
This tool never rewrites inputs and refuses to overwrite an existing report.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def read_steps(directory):
    return [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()
            if line.strip()]


def tensor_differences(left, right):
    """Report exact equality, absolute error, L2, and RMS over matched tensors."""
    if tuple(left) != tuple(right):
        raise ValueError('Tensor names/order differ')
    largest_error = squared_error = 0.0
    elements = changed = 0
    none_mismatches = []
    for name, value in left.items():
        other = right[name]
        if value is None or other is None:
            if value is not other:
                none_mismatches.append(name)
            continue
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Tensor shape/dtype differs: ' + name)
        if value.is_floating_point():
            finite = torch.isfinite(value) & torch.isfinite(other)
            same_special = (value == other) | (torch.isnan(value) & torch.isnan(other))
            if not bool((finite | same_special).all()):
                raise ValueError('Unmatched non-finite values: ' + name)
            difference = value[finite].double() - other[finite].double()
        else:
            difference = value.double() - other.double()
        elements += difference.numel()
        if difference.numel():
            largest_error = max(largest_error, float(difference.abs().max()))
            squared_error += float(difference.square().sum())
            changed += int(torch.count_nonzero(difference))
    return dict(exact=changed == 0 and not none_mismatches,
                max_abs=largest_error, l2=math.sqrt(squared_error),
                rms=math.sqrt(squared_error / max(elements, 1)),
                elements=elements, changed_elements=changed,
                none_mismatches=none_mismatches)


def update_statistics(previous, current, gradients):
    """Measure actual parameter changes, excluding frozen buffers."""
    squared_update = squared_parameter = 0.0
    elements = changed = 0
    for name, gradient in gradients.items():
        if gradient is None:
            continue
        old, new = previous[name].double(), current[name].double()
        difference = new - old
        squared_update += float(difference.square().sum())
        squared_parameter += float(old.square().sum())
        elements += difference.numel()
        changed += int(torch.count_nonzero(difference))
    return dict(l2=math.sqrt(squared_update),
                rms=math.sqrt(squared_update / max(elements, 1)),
                relative_l2=math.sqrt(squared_update / max(squared_parameter, 1e-30)),
                changed_elements=changed, elements=elements)


def step_loss_lines(directory):
    paths = list((directory / 'log').rglob('*.txt'))
    if len(paths) != 1:
        raise ValueError('Expected one private training text log')
    return [line for line in paths[0].read_text().splitlines() if line.startswith('StepLoss ')]


def compare_runs(before, after):
    left_steps, right_steps = read_steps(before), read_steps(after)
    if not left_steps or len(left_steps) != len(right_steps):
        raise ValueError('Missing or mismatched completed steps')
    previous_left = torch.load(before / 'initial.pth', map_location='cpu')
    previous_right = torch.load(after / 'initial.pth', map_location='cpu')
    report = dict(before=str(before), after=str(after),
                  initial_model=tensor_differences(previous_left, previous_right),
                  step_loss_lines_equal=step_loss_lines(before) == step_loss_lines(after), steps=[])
    for left, right in zip(left_steps, right_steps):
        if left['step'] != right['step']:
            raise ValueError('Step numbers differ')
        step = left['step']
        left_state = torch.load(before / ('step_%03d.pth' % step), map_location='cpu')
        right_state = torch.load(after / ('step_%03d.pth' % step), map_location='cpu')
        report['steps'].append(dict(
            step=step,
            actual_before=left['policy']['current_compression_value'],
            actual_after=right['policy']['current_compression_value'],
            geometry_before=left['policy']['current_geometry_value'],
            geometry_after=right['policy']['current_geometry_value'],
            counts_before=left['plan']['selected_counts'], counts_after=right['plan']['selected_counts'],
            plan_equal=left['plan'] == right['plan'], policy_equal=left['policy'] == right['policy'],
            native_metrics_equal=[call['metrics'] for call in left['native_calls']]
                                 == [call['metrics'] for call in right['native_calls']],
            stream_bits_equal=[[stream['bits'] for stream in call['streams']] for call in left['native_calls']]
                              == [[stream['bits'] for stream in call['streams']] for call in right['native_calls']],
            stream_hashes_equal=[[stream['sha256'] for stream in call['streams']] for call in left['native_calls']]
                                == [[stream['sha256'] for stream in call['streams']] for call in right['native_calls']],
            gradients=tensor_differences(left_state['gradients'], right_state['gradients']),
            model=tensor_differences(left_state['model'], right_state['model']),
            update_before=update_statistics(previous_left, left_state['model'], left_state['gradients']),
            update_after=update_statistics(previous_right, right_state['model'], right_state['gradients'])))
        previous_left, previous_right = left_state['model'], right_state['model']
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--unchanged-repeat', type=Path,
                        help='Optional second execution of the original source, for a noise control')
    parser.add_argument('--live-steps', type=Path,
                        help='Read-only extracted AutonomousLearningAudit rows from the running log')
    parser.add_argument('--out', required=True, type=Path)
    options = parser.parse_args()
    if options.out.exists():
        raise FileExistsError(options.out)
    result = dict(refactor=compare_runs(options.before, options.after))
    if options.unchanged_repeat:
        result['unchanged_source_repeat'] = compare_runs(options.unchanged_repeat, options.before)
    if options.live_steps:
        live_steps = json.loads(options.live_steps.read_text())
        after_steps = read_steps(options.after)
        result['live_log_matches'] = []
        for live, measured in zip(live_steps, after_steps):
            result['live_log_matches'].append(dict(
                step=measured['step'],
                actual_equal=live['policy']['current_compression_value']
                             == measured['policy']['current_compression_value'],
                geometry_equal=live['policy']['current_geometry_value']
                               == measured['policy']['current_geometry_value'],
                counts_equal=live['counts'] == measured['plan']['selected_counts'],
                ratios_equal=live['ratios'] == measured['plan']['selected_amount_ratios']))
    with options.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('Comparison saved:', options.out)
    print('Exact scalar equality is distinct from bitwise gradient/stream equality; see all fields.')


if __name__ == '__main__':
    main()
