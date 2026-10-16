"""Run the frozen trainer with a checked sidecar/teacher compatibility adapter."""
import argparse
import os
from pathlib import Path
import statistics
import sys

from protocol import BASE_SEED, KIND, RUNTIME, SETTINGS, SMOKE_STEPS, instrument, read, sha

APP = Path(os.environ.get('COTRACKER_APP_DIR', '/opt/app'))
sys.path[:0] = [str(APP / 'ext/co-tracker'), str(APP), str(APP / 'experiments')]


def run(root, seed, policy, smoke):
    import torch
    from cotracker.models.core.cotracker.backbone_growth import save_json
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One visible CUDA GPU required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Only single-device training is paired')
    config = read(root / 'run-config.json')
    if config['sidecar_runtime'] != RUNTIME or config['training_seed'] != BASE_SEED + seed or config['policy'] != policy:
        raise ValueError('Wrong frozen arm config')
    if config['steps'] != (SMOKE_STEPS if smoke else 1000) or config['kind'] != KIND:
        raise ValueError('Wrong stage/sidecar')
    runtime_path = Path('/study/runtime.json')
    if read(runtime_path) != RUNTIME:
        raise ValueError('Training runtime changed')
    trainer = APP / 'ext/co-tracker/train_on_real_data.py'
    manifest = read(Path('/study/manifest.json'))
    if sha(trainer) != manifest['trainer_sha256'] or sha(Path('/study/manifest.json')) != config['manifest_sha256']:
        raise ValueError('Frozen source/protocol changed')
    source = instrument(trainer.read_text(encoding='utf-8'))
    save_json(root / 'trainer-adapter.json', {'source_sha256': sha(trainer),
              'adapter_sha256': sha(Path(__file__)), 'protocol_sha256': sha(Path(__file__).with_name('protocol.py')),
              'stop_steps': config['steps'], 'scheduler_steps': 1000})
    elapsed = []

    def health(row):
        free, _ = torch.cuda.mem_get_info()
        row['health'] = {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                         'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                         'driver_free_mib': free / 2**20}
        elapsed.append(row['step_seconds'])
        h = row['health']
        save_json(root / 'latest-health.json', {'step': row['step'], 'seconds': row['step_seconds'], **h})
        if max(h['peak_allocated_mib'], h['peak_reserved_mib']) > 20480 or free < 2 * 2**30:
            raise RuntimeError('GPU headroom exceeded; preserve checkpoints and stop')
        if len(elapsed) >= 7 and statistics.median(elapsed[-5:]) > 8:
            raise RuntimeError('Median of five steps exceeded 8 seconds; stop before another long slow run')

    checkpoint = SETTINGS[policy][0]
    training_seed = str(BASE_SEED + seed)
    argv = ['--batch_size', '1', '--num_workers', '0', '--num_steps', '1000',
            '--ckpt_path', str(root), '--model_name', 'cotracker_three',
            '--experiment_name', f'frozen_mamba_seed_{seed}_{policy}',
            '--sequence_len', '60', '--traj_per_sample', '128', '--train_iters', '4',
            '--save_freq', '100000', '--save_every_n_steps', '1' if smoke else '25',
            '--keep_last_checkpoints', '3', '--save_every_n_epoch', '1',
            '--trackrad_data_dir', '/data', '--skip_evaluation', '--offline_model',
            '--restore_ckpt', '/checkpoints/scaled_offline.pth',
            '--teacher_types', 'cotracker2v1', 'online_cotracker_three', 'offline_cotracker_three',
            '--teacher_cotracker2_ckpt', '/checkpoints/cotracker2v1.pth',
            '--teacher_online_ckpt', '/checkpoints/baseline_online.pth',
            '--teacher_offline_ckpt', '/checkpoints/' + checkpoint, '--auxiliary_teacher_weight', '0',
            '--seed', training_seed, '--paired_step_seed', training_seed, '--teacher_seed', training_seed,
            '--teacher_log_every', '1', '--confidence_target_mode', 'hard', '--lr', '0.00005',
            '--backbone_growth_study', '--backbone_architecture', 'base', '--feature_sidecar', KIND,
            '--teacher_student_study', '--offline_teacher_policy', policy]
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(runtime_path)
    os.chdir(trainer.parent)
    sys.argv = [str(trainer)] + argv
    exec(compile(source, str(trainer), 'exec'), {'__name__': '__main__', '__file__': str(trainer),
         '__stop_steps__': config['steps'], '__health__': health})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/results'))
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--policy', choices=tuple(SETTINGS), required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(a.root, a.seed, a.policy, a.smoke)
