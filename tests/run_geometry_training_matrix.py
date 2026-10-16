"""Sequential isolated codec smoke runs; never controls an existing process."""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--only', nargs='*', default=[])
    parser.add_argument('--suffix', default='')
    opts = parser.parse_args()
    root = os.path.abspath(opts.root)
    common = ['--max_files', '1', '--exploration_schedule_episodes', '382',
              '--more_training', 'false']
    cases = [
        ('gpcc-legacy-control', 'G-PCC', ['--geometry_octree_joint_policy', 'false']),
        ('gpcc-final-verified', 'G-PCC', []),
        ('octattention-final-verified', 'OctAttention', ['--octattention_teacher_device', 'cuda']),
        ('sparse-after', 'SparsePCGC', []),
    ]
    for name, codec, extra in cases:
        if opts.only and name not in opts.only:
            continue
        case_root = os.path.join(root, name + ('-' + opts.suffix if opts.suffix else ''))
        os.makedirs(case_root, exist_ok=True)
        isolated = ['--log_root', case_root+'/log', '--out_path', case_root+'/checkpoints',
                    '--surrogate_pretrained_root', case_root+'/surrogate']
        if codec == 'SparsePCGC':
            script = 'train.py'
            detail = ['--episodes', '1']
        else:
            script = 'tests/run_geometry_training_smoke.py'
            detail = ['--episodes', '2', '--train_frames_per_sequence', '1',
                      '--input_dir', root+'/fixture/ground', '--train_full_cloud_val_frames', '1',
                      '--heuristic_guidance_online_grad_audit', 'true',
                      '--heuristic_guidance_online_source_grad_probe_interval', '1']
        command = [sys.executable, '-B', script, '--compression', codec] + common + detail + isolated + extra
        print('START', name, flush=True)
        with open(case_root+'/stdout.txt', 'w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        print('COMPLETE', name, flush=True)


if __name__ == '__main__':
    main()
