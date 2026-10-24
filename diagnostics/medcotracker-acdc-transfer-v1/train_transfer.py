"""Reuse the audited TrackRAD trainer, replacing only checked initialization/teacher dispatch."""
import argparse
import statistics
from common import *


def initialize(model, root, phase, arm):
    from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, configure_sidecar_runtime, set_sidecar_training_mode
    proof = marker(root, f'{phase}/init/{arm}.json')
    if proof is None:
        raise ValueError('Missing audited initialization')
    source = root / phase / 'init' / (arm + '.pth')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    attach_sidecar(model, 'mamba')
    model.load_state_dict(saved['model'], strict=True)
    configure_sidecar_runtime(model, **RUNTIME)
    set_sidecar_training_mode(model)
    if S.state_hashes(model) != proof['state_hashes']:
        raise ValueError('Strict handoff changed tensors')
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith(PREFIX) for n in trainable):
        raise ValueError('TrackRAD must train only Mamba')
    return {'architecture': 'base', 'arm': arm, 'phase': phase, 'init_sha256': digest(source),
            'strict_handoff_exact': True, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'source_sha256': proof['source_sha256'], 'A1_carried_branch': arm == 'A1'}


def run(root, phase, arm, folder, data, weights):
    gpu()
    plan = checked_plan(root)
    verify_files(weights, plan['weights_sha256'])
    verify_dataset(data, plan['trackrad_files']['train-40'], plan['trackrad_ids']['train-40'])
    cfg = config(root, phase, arm)
    S.frozen(folder / 'run-config.json', cfg)
    if phase == 'pilot' and marker(root, 'smoke/verified.json') is None:
        raise ValueError('Missing transfer smoke gate')
    if (folder / 'cotracker_three_final.pth').exists():
        from study import verify_arm
        verify_arm(root, phase, arm)
        print('CACHE VERIFIED trained ' + phase + '/' + arm, flush=True)
        return
    trainer = APP / 'ext/co-tracker/train_on_real_data.py'
    source = instrument(trainer.read_text(encoding='utf-8'))
    S.frozen(folder / 'trainer-adapter.json', {'source_sha256': digest(trainer),
             'executed_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
             'stop_steps': cfg['steps'], 'scheduler_steps': 1000, 'init_sha256': cfg['init_sha256']})
    elapsed = []

    def health(row):
        free, _ = torch.cuda.mem_get_info()
        row['health'] = {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                         'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                         'driver_free_mib': free / 2**20}
        if row['label_audit']['pool_sha256'] != cfg['pool_sha256']:
            raise ValueError('Teacher pool changed')
        elapsed.append(row['step_seconds'])
        save(folder / 'latest-health.json', {'step': row['step'], 'seconds': row['step_seconds'], **row['health']})
        if max(row['health']['peak_allocated_mib'], row['health']['peak_reserved_mib']) > 20480 or free < 2 * 2**30:
            raise RuntimeError('GPU headroom limit; preserve checkpoint')
        if len(elapsed) >= 7 and statistics.median(elapsed[-5:]) > 8:
            raise RuntimeError('Transfer median >8s; inspect before larger training')

    def dispatch(slot, teacher_type, teacher, video, queries, draw, policy):
        if policy != arm: raise ValueError('Wrong transfer arm')
        return T.teacher_forward(slot, teacher_type, teacher, video, queries, draw, arm, cfg['pool'])

    seed = str(S.STUDENT_SEED)
    argv = ['--batch_size', '1', '--num_workers', '0', '--num_steps', '1000',
            '--ckpt_path', str(folder), '--model_name', 'cotracker_three',
            '--experiment_name', f'medcotracker_{phase}_seed_0_{arm}',
            '--sequence_len', '60', '--traj_per_sample', '128', '--train_iters', '4',
            '--save_freq', '100000', '--save_every_n_steps', '1' if phase == 'smoke' else '25',
            '--keep_last_checkpoints', '3', '--save_every_n_epoch', '1',
            '--trackrad_data_dir', str(data), '--skip_evaluation', '--offline_model',
            '--restore_ckpt', str(weights / 'scaled_offline.pth'),
            '--teacher_types', 'cotracker2v1', 'online_cotracker_three', 'offline_cotracker_three',
            '--teacher_cotracker2_ckpt', str(weights / 'cotracker2v1.pth'),
            '--teacher_online_ckpt', str(weights / 'baseline_online.pth'),
            '--teacher_offline_ckpt', str(weights / 'scaled_offline.pth'),
            '--train_grid_size', '5', '--train_sift_size', '0', '--auxiliary_teacher_weight', '0',
            '--seed', seed, '--paired_step_seed', seed, '--teacher_seed', seed,
            '--teacher_log_every', '1', '--confidence_target_mode', 'hard', '--lr', '0.00005',
            '--backbone_growth_study', '--backbone_architecture', 'base', '--feature_sidecar', 'mamba',
            '--teacher_student_study', '--offline_teacher_policy', arm]
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(root / 'runtime.json')
    os.chdir(trainer.parent)
    sys.argv = [str(trainer)] + argv
    exec(compile(source, str(trainer), 'exec'), {'__name__': '__main__', '__file__': str(trainer),
         '__stop_steps__': cfg['steps'], '__health__': health, '__dispatch__': dispatch,
         '__settings__': {arm: 'scaled_offline.pth'}, '__pool_sha__': cfg['pool_sha256'],
         '__initialize__': lambda model, args: initialize(model, root, phase, arm)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('smoke', 'pilot'), required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    a = p.parse_args()
    root = Path('/study')
    run(root, a.phase, a.arm, root / a.phase / 'train' / a.arm, Path('/data'), Path('/weights'))
