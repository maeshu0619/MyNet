"""Configure isolated, matched runs using the existing gradient-audit runner."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument('--autonomous', choices=['on', 'off'], default='off')
    parser.add_argument('--diagnostic-frames', type=int, default=4)
    parser.add_argument('--reference-json', type=Path)
    parser.add_argument('--repeat-validation', action='store_true')
    parser.add_argument('--source-snapshot', type=Path,
                        help='Read-only replay of the restored original source (autonomous off only)')
    options, remainder = parser.parse_known_args()
    if options.source_snapshot:
        if options.autonomous != 'off':
            parser.error('--source-snapshot replays only the original den6 route')
        from experiments.sparsepcgc_direction.common import FrozenSources
        sys.meta_path.insert(0, FrozenSources(options.source_snapshot.resolve()))
    output = Path(remainder[remainder.index('--out') + 1]).resolve()
    # Private copy: the audit must never update the production registry.
    registry = output.parent / (output.name + '_surrogate_registry')
    registry.mkdir(exist_ok=False)
    source = Path('/data/maejima/data/pretrained_surrogate/SparsePCGC/20260729_001750.pth')
    if source.is_file():
        (registry / 'SparsePCGC').mkdir()
        shutil.copy2(str(source), str(registry / 'SparsePCGC' / source.name))
    import train as training
    original = training.parse_pugan_args
    original_network = training.Network
    original_loss = training.Loss
    models = {}

    def network(*args, **kwargs):
        value = original_network(*args, **kwargs)
        models['value'] = value
        original_nodes = value._build_node_voxel_input
        node_rows = {}

        def nodes(*positional, **named):
            result = original_nodes(*positional, **named)
            node_rows['coords'] = result[0]['voxel_coords'] if result[0] is not None else None
            return result

        def alignment(module, inputs, result):
            if value.training:
                return
            expected, actual = node_rows.get('coords'), result.get('global_voxel_coords')
            if expected is not None and actual is not None and expected.shape == actual.shape:
                fraction = float((expected == actual).all(dim=1).float().mean().cpu())
                with (output/'validation_alignment.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(frame=str(getattr(value.args, '_current_input_file', '')),
                        clock=int(getattr(value.args, '_global_train_step', 0)),
                        node_structure_row_match=fraction))+'\n')

        value._build_node_voxel_input = nodes
        value.structure_analyzer.register_forward_hook(alignment)
        original_forward = value.forward

        def observed_forward(*positional, **named):
            result = original_forward(*positional, **named)
            debug = value.last_actuator_voxel_state.get('ana_den6_exact_residual_plan_debug', {})
            if debug.get('autonomous_joint_policy'):
                keys = ('selected_counts', 'selected_amount_ratios', 'requested_counts',
                        'joint_amount_bins_by_operation', 'joint_requested_ratios',
                        'joint_where_selection_change_rates', 'joint_where_attempted_counts',
                        'candidate_pool_sizes', 'joint_max_per_operation', 'exploration_multiplier',
                        'joint_amount_log_prob', 'joint_where_log_prob', 'final_edited_voxel_hash')
                row = {key: debug.get(key) for key in keys}
                row.update(training=bool(value.training),
                           global_step=int(getattr(value.args, '_global_train_step', 0)),
                           frame=str(getattr(value.args, '_current_input_file', '')))
                with (output/'joint_decisions.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            return result

        value.forward = observed_forward
        return value

    def loss(*args, **kwargs):
        value = original_loss(*args, **kwargs)
        geometry = value.get_geometry_loss

        def measured(settings, *positional, **named):
            result = geometry(settings, *positional, **named)
            if 'value' in models and not models['value'].training:
                from models.utils.training.sparsepcgc_autonomous_policy import validation_fingerprint
                row = dict(validation_fingerprint(settings, value, getattr(settings, '_current_input_file', '')),
                           clock=int(getattr(settings, '_global_train_step', 0)),
                           geometry=float(result.detach().cpu()))
                with (output/'validation_geometry.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            return result
        value.get_geometry_loss = measured
        return value

    training.Network = network
    training.Loss = loss
    if options.repeat_validation:
        from models.utils.training import train_runtime as runtime
        original_validation = runtime.run_episode_full_cloud_validation

        def validation(*positional, **named):
            first = original_validation(*positional, **named)
            second = original_validation(*positional, **named)
            checks = {key: first.get(key) == second.get(key) for key in (
                'value', 'geometry_value', 'count', 'sample_signature')}
            with (output/'repeat_validation.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(first=first, second=second, identical=checks))+'\n')
            if not all(checks.values()):
                raise RuntimeError('Repeated deterministic validation changed')
            return first

        runtime.run_episode_full_cloud_validation = validation

    def parse(*args, **kwargs):
        extra = ['--train_fixed_episode_frames', str(options.diagnostic_frames),
                 '--surrogate_pretrained_root', str(registry), '--more_training', 'false']
        if not options.source_snapshot:
            extra += ['--test_route', 'true' if options.autonomous == 'on' else 'false']
        if options.reference_json:
            extra += ['--sparsepcgc_autonomous_reference_json', str(options.reference_json.resolve())]
        sys.argv += extra
        return original(*args, **kwargs)

    training.parse_pugan_args = parse
    from experiments.sparsepcgc_direction.run import main as run
    sys.argv = [sys.argv[0]] + remainder
    run()
    from experiments.summarize_autonomous_search import rows, measured_validation
    clocks = sorted({row['clock'] for row in rows(output/'validation_geometry.jsonl')})
    for clock in clocks:
        with (output/('validation_reference_%03d.json' % clock)).open('x') as stream:
            json.dump(dict(measurements=measured_validation(output, clock),
                           source_run=str(output), scope='fixed development validation, not a full test-set guarantee'),
                      stream, indent=2)
    if options.autonomous == 'on':
        from models.utils.training.sparsepcgc_autonomous_policy import autonomous_checkpoint_status
        (output/'promotion_status.json').write_text(json.dumps(
            autonomous_checkpoint_status(models['value'].args), indent=2))


if __name__ == '__main__':
    main()
