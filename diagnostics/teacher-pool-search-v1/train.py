"""Fixed Mamba student, frozen backbone, configurable immutable teacher pool."""
import argparse
import os
from pathlib import Path
import statistics
import sys

from protocol import BASE_SEED, RUNTIME, config, digest, instrument, read, sha, teacher_forward
from shared import APP, legacy, legacy_protocol


def run(root, study, phase, seed, arm):
    torch = legacy.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('One visible CUDA GPU and one training process required')
    cfg = read(root / 'run-config.json')
    if cfg != config(seed, arm, phase, cfg['pool'], sha(study / 'manifest.json')):
        raise ValueError('Wrong frozen arm configuration')
    if read(study / 'runtime.json') != RUNTIME:
        raise ValueError('Runtime configuration changed')
    manifest = read(study / 'manifest.json')
    package = Path(__file__).parent
    for name, value in manifest['package_sha256'].items():
        if sha(package / name) != value:
            raise ValueError('Search package changed')
    trainer = APP / 'ext/co-tracker/train_on_real_data.py'
    if sha(trainer) != manifest['trainer_sha256']:
        raise ValueError('Frozen trainer source changed')
    source = instrument(trainer.read_text(encoding='utf-8'), legacy_protocol.instrument)
    legacy.frozen(root / 'trainer-adapter.json', {'source_sha256': sha(trainer),
                  'adapter_sha256': sha(Path(__file__)), 'protocol_sha256': sha(package / 'protocol.py'),
                  'executed_source_sha256': __import__('hashlib').sha256(source.encode()).hexdigest(),
                  'pool_sha256': cfg['pool_sha256'], 'stop_steps': cfg['steps'], 'scheduler_steps': 1000})
    elapsed = []

    def health(row):
        free, _ = torch.cuda.mem_get_info()
        row['health'] = {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                         'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20, 'driver_free_mib': free / 2**20}
        if row['label_audit']['pool_sha256'] != digest(cfg['pool']):
            raise ValueError('Teacher pool changed during training')
        elapsed.append(row['step_seconds'])
        h = row['health']
        legacy.save_json(root / 'latest-health.json', {'step': row['step'], 'seconds': row['step_seconds'], **h})
        if max(h['peak_allocated_mib'], h['peak_reserved_mib']) > 20480 or free < 2 * 2**30:
            raise RuntimeError('GPU headroom exceeded; preserve results')
        if len(elapsed) >= 7 and statistics.median(elapsed[-5:]) > 8:
            raise RuntimeError('Training median exceeds 8 seconds; stop before a long slow queue')

    def dispatch(slot, teacher_type, model, video, queries, draw, policy):
        if policy != arm:
            raise ValueError('Wrong teacher arm')
        return teacher_forward(slot, teacher_type, model, video, queries, draw, arm, cfg['pool'])

    training_seed = str(BASE_SEED + seed)
    argv = ['--batch_size', '1', '--num_workers', '0', '--num_steps', '1000',
            '--ckpt_path', str(root), '--model_name', 'cotracker_three',
            '--experiment_name', f'teacher_search_{phase}_seed_{seed}_{arm}',
            '--sequence_len', '60', '--traj_per_sample', '128', '--train_iters', '4',
            '--save_freq', '100000', '--save_every_n_steps', '1' if phase == 'smoke' else '25',
            '--keep_last_checkpoints', '3', '--save_every_n_epoch', '1',
            '--trackrad_data_dir', '/data', '--skip_evaluation', '--offline_model',
            '--restore_ckpt', '/checkpoints/scaled_offline.pth',
            '--teacher_types', 'cotracker2v1', 'online_cotracker_three', 'offline_cotracker_three',
            '--teacher_cotracker2_ckpt', '/checkpoints/cotracker2v1.pth',
            '--teacher_online_ckpt', '/checkpoints/baseline_online.pth',
            '--teacher_offline_ckpt', '/checkpoints/' + cfg['pool'][2]['checkpoint'],
            '--train_grid_size', '5', '--train_sift_size', '0', '--auxiliary_teacher_weight', '0',
            '--seed', training_seed, '--paired_step_seed', training_seed, '--teacher_seed', training_seed,
            '--teacher_log_every', '1', '--confidence_target_mode', 'hard', '--lr', '0.00005',
            '--backbone_growth_study', '--backbone_architecture', 'base', '--feature_sidecar', 'mamba',
            '--teacher_student_study', '--offline_teacher_policy', arm]
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(study / 'runtime.json')
    os.chdir(trainer.parent)
    sys.argv = [str(trainer)] + argv
    exec(compile(source, str(trainer), 'exec'), {'__name__': '__main__', '__file__': str(trainer),
         '__stop_steps__': cfg['steps'], '__health__': health, '__dispatch__': dispatch,
         '__settings__': {arm: cfg['pool'][2]['checkpoint']}, '__pool_sha__': cfg['pool_sha256']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/results'))
    p.add_argument('--study', type=Path, default=Path('/study'))
    p.add_argument('--phase', choices=('smoke', 'pilot', 'full'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--arm', required=True)
    a = p.parse_args()
    run(a.root, a.study, a.phase, a.seed, a.arm)
