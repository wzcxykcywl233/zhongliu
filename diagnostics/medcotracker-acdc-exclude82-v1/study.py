"""Initialization, frozen-tensor audits, pinned runtime smoke, and official validation."""
import argparse
from common import *


def handoff(root, phase, arm, weights, external_smoke):
    plan = checked_plan(root)
    verify_files(weights, plan['weights_sha256'])
    done = f'{phase}/init/{arm}.json'
    if marker(root, done):
        print('CACHE VERIFIED handoff ' + phase + '/' + arm, flush=True)
        return
    if phase == 'full' and marker(root, 'smoke/verified.json') is None:
        raise ValueError('Missing smoke gate')
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    source = weights / 'scaled_offline.pth'
    if arm == 'R1':
        model = build_cotracker(str(source), offline=True, window_len=60)
    else:
        if phase == 'smoke':
            if digest(external_smoke / 'summary.json') != plan['external_smoke_summary_sha256']:
                raise ValueError('External smoke changed')
            report = S.load_verified(external_smoke, arm + '-verified.json')
            paths = [p for p in report['files_sha256'] if p.endswith('/external-smoke.pth')]
            if len(paths) != 1:
                raise ValueError('Ambiguous external smoke checkpoint')
            source = S.safe_file(external_smoke, paths[0])
            expected_steps = 5
        else:
            if marker(root, f'full/external/{arm}/verified.json') is None:
                raise ValueError('External full incomplete')
            source = root / 'full/external' / arm / 'external-final.pth'
            expected_steps = RULE['external_steps']
        saved = torch.load(source, map_location='cpu', weights_only=True)
        if saved['route'] != arm or saved['steps'] != expected_steps:
            raise ValueError('Wrong external handoff')
        model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        S.configure(model, arm)
        model.load_state_dict(saved['model'], strict=True)
    source_state = S.state_hashes(model)
    proof = S.configure_handoff(model, arm)
    final_state = S.state_hashes(model)
    if any(final_state[n] != h for n, h in source_state.items()):
        raise ValueError('Handoff altered imported tensors')
    path = root / phase / 'init' / (arm + '.pth')
    atomic_torch(path, {'model': model.state_dict(), 'backbone_architecture': 'base', 'feature_sidecar': 'mamba',
                         'arm': arm, 'phase': phase, 'source_sha256': digest(source),
                         'manifest_sha256': digest(root / 'manifest.json')})
    publish(root, done, [path], arm=arm, phase=phase, source_sha256=digest(source),
            state_hashes=final_state, **proof)
    folder = root / phase / 'train' / arm
    folder.mkdir(parents=True, exist_ok=True)
    S.frozen(folder / 'run-config.json', config(root, phase, arm))
    print('HANDOFF VERIFIED ' + phase + '/' + arm, flush=True)


def verify_arm(root, phase, arm):
    plan = checked_plan(root)
    initial = marker(root, f'{phase}/init/{arm}.json')
    if initial is None:
        raise ValueError('Missing initialization proof')
    folder = root / phase / 'train' / arm
    cfg = config(root, phase, arm)
    if read(folder / 'run-config.json') != cfg:
        raise ValueError('Changed arm configuration')
    trainer = APP / 'ext/co-tracker/train_on_real_data.py'
    expected_adapter = {'source_sha256': digest(trainer),
                        'executed_source_sha256': hashlib.sha256(instrument(trainer.read_text(encoding='utf-8')).encode()).hexdigest(),
                        'stop_steps': cfg['steps'], 'scheduler_steps': 1000, 'init_sha256': cfg['init_sha256']}
    if read(folder / 'trainer-adapter.json') != expected_adapter:
        raise ValueError('Executed trainer adapter changed')
    final = folder / 'cotracker_three_final.pth'
    saved = torch.load(final, map_location='cpu', weights_only=True)
    if (saved['total_steps'] != cfg['steps'] or saved['feature_sidecar'] != 'mamba' or
            saved['backbone_architecture'] != 'base' or saved['teacher_student_policy'] != arm or
            saved['teacher_pool_sha256'] != cfg['pool_sha256']):
        raise ValueError('Wrong trained checkpoint')
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar
    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    attach_sidecar(model, 'mamba')
    model.load_state_dict(saved['model'], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite transfer weights')
    after = S.state_hashes(model)
    before = initial['state_hashes']
    if set(after) != set(before) or any(after[n] != h for n, h in before.items() if not n.startswith(PREFIX)):
        raise ValueError('Frozen backbone/buffers changed during TrackRAD transfer')
    if not any(after[n] != h for n, h in before.items() if n.startswith(PREFIX)):
        raise ValueError('Sidecar did not learn')
    init_report = read(folder / 'initialization-report.json')
    if (init_report['init_sha256'] != cfg['init_sha256'] or
            saved['architecture_report'] != init_report or not init_report['strict_handoff_exact']):
        raise ValueError('Wrong actual trainer initialization')
    rows = T.records(folder / 'paired-steps.jsonl', cfg)
    if any(not set(r['case_names']).issubset(plan['trackrad_ids']['train-40']) for r in rows.values()):
        raise ValueError('Transfer training used an unauthorized case')
    if {r['primary_index'] for r in rows.values()} != {0, 1, 2}:
        raise ValueError('Not all teacher slots exercised')
    publish(root, f'{phase}/train/{arm}/verified.json',
            [final, folder / 'run-config.json', folder / 'paired-steps.jsonl',
             folder / 'initialization-report.json', folder / 'trainer-adapter.json'],
            steps=cfg['steps'], arm=arm, frozen_backbone_exact=True, strict_reload=True)
    print('AUDIT PASSED ' + phase + '/' + arm, flush=True)
    return rows


def audit(root, phase):
    rows = {arm: verify_arm(root, phase, arm) for arm in ARMS}
    for arm in ARMS[1:]:
        T.paired(rows['R1'], rows[arm], T.baseline(), T.baseline())
    fresh = [marker(root, f'{phase}/init/{a}.json')['sidecar_hashes'] for a in ARMS]
    if any(value != fresh[0] for value in fresh):
        raise ValueError('Fresh sidecar initializations not paired')
    if phase == 'full':
        signatures = []
        for arm in ('B1',):
            proof = marker(root, f'full/external/{arm}/verified.json')
            if proof is None or proof['steps'] != RULE['external_steps']:
                raise ValueError('External stage incomplete')
            records = read(root / 'full/external' / arm / 'steps.json')
            if [r['step'] for r in records] != list(range(RULE['external_steps'])):
                raise ValueError('Missing external steps')
            planned = read(root / 'exclusion-schedule.json')['rows']
            if len(records) != len(planned) or any((r['patient'], r['slot']) != (p['patient'], p['slot']) for r, p in zip(records, planned)):
                raise ValueError('External training violated patient082 exclusion schedule')
            signatures.append([(r['patient'], r['slot'], r['frames'], r['label_sha256']) for r in records])
        if any(x != signatures[0] for x in signatures):
            raise ValueError('External clips or labels not paired')
        for step, (_, _, _, expected) in enumerate(signatures[0]):
            if digest(root / 'full/labels' / f'{step:04}.pth') != expected:
                raise ValueError('External label cache changed')
    files = [root / phase / 'train' / arm / name for arm in ARMS
             for name in ('verified.json', 'cotracker_three_final.pth', 'paired-steps.jsonl')]
    files += [root / phase / 'init' / (arm + suffix) for arm in ARMS for suffix in ('.json', '.pth')]
    if phase == 'full':
        files += [root / 'full/external/B1' / name for name in
                  ('verified.json', 'external-final.pth', 'steps.json', 'initial.json')]
        files += [root / 'exclusion-schedule.json', root / 'smoke/exclusion-verified.json']
    publish(root, f'{phase}/training-audit.json', files, paired_inputs_and_labels=True,
            frozen_backbones_exact=True, fresh_B_and_R1_branches_exact=True, arms=list(ARMS))
    print('AUDIT PASSED paired ' + phase, flush=True)


def runtime(root):
    plan = checked_plan(root)
    included = {}
    prefix = 'cotracker-algorithm/'
    for relative, expected in plan['source_sha256'].items():
        if not relative.startswith(prefix) or not relative.endswith('.py'):
            continue
        inner = relative[len(prefix):]
        if '/' in inner and not inner.startswith(('ext/co-tracker/', 'resources/', 'experiments/')):
            continue
        path = Path('/opt/app') / inner
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Pinned inference source missing/changed: ' + relative)
        included[relative] = expected
    if not included:
        raise ValueError('Empty inference inventory')
    S.frozen(root / 'inference-runtime.json', {'image': plan['images']['Inference'], 'source_sha256': included})
    print('RUNTIME VERIFIED Inference', flush=True)


def inference_smoke(root):
    gpu()
    runtime(root)
    if marker(root, 'smoke/training-audit.json') is None:
        raise ValueError('Missing paired transfer smoke audit')
    # Exercise the actual pinned image entry/loader, not the checkout's import path.
    sys.path[:0] = ['/opt/app/ext/co-tracker', '/opt/app']
    import model as entry
    import cotracker
    if Path('/opt/app') not in Path(cotracker.__file__).resolve().parents:
        raise ValueError('Inference smoke imported wrong CoTracker source')
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(root / 'runtime.json')
    reset_rng(S.STUDENT_SEED)
    video = torch.rand(1, 60, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [29., 32., 32.], [59., 48., 48.]]], device='cuda')
    proofs = []
    for arm in ARMS:
        model = entry.resources.setup_model(str(root / 'smoke/train' / arm / 'cotracker_three_final.pth'), device='cuda')
        d = {}
        model.updateformer.feature_sidecar.diagnostics = d
        with torch.no_grad():
            output = model(video=video, queries=queries, iters=2)
        if (not model.frozen_pretrained or model.feature_sidecar_kind != 'mamba' or
                model.updateformer.feature_sidecar.runtime != RUNTIME or d.get('sidecar_calls') != 2 or
                not d.get('sidecar_residual_rms_max') or any(not bool(torch.isfinite(t).all()) for t in output[:3])):
            raise ValueError('Trained transfer inference smoke failed: ' + arm)
        proofs.append({'arm': arm, **d})
        del model
        torch.cuda.empty_cache()
    save(root / 'smoke/inference.json', {'proofs': proofs, 'note': 'Synthetic loader/forward check, not accuracy.'})
    files = [root / 'smoke/training-audit.json', root / 'smoke/inference.json', root / 'inference-runtime.json']
    files += [root / 'smoke/train' / arm / 'cotracker_three_final.pth' for arm in ARMS]
    publish(root, 'smoke/verified.json', files, stage='TrackRAD-transfer-smoke',
            steps_per_arm=RULE['smoke_steps'], seed=SEED_INDEX)
    print('GPU TRANSFER SMOKE PASSED R1/B1', flush=True)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('plan', 'handoff', 'audit', 'runtime', 'inference-smoke', 'gate'))
    p.add_argument('--phase', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--arm', choices=ARMS, default='R1')
    a = p.parse_args()
    root = Path('/study')
    if a.command == 'plan':
        prepare(root, Path('/prior'), Path('/audit'), Path('/external-smoke'), Path('/weights'),
                Path('/datasets/train'), Path('/datasets/validation'), read(Path('/prior/local-runtime-images.json')),
                Path('/pilot-source'))
    elif a.command == 'handoff': handoff(root, a.phase, a.arm, Path('/weights'), Path('/external-smoke'))
    elif a.command == 'audit': audit(root, a.phase)
    elif a.command == 'runtime': runtime(root)
    elif a.command == 'inference-smoke': inference_smoke(root)
    else:
        checked_plan(root)
        if marker(root, 'smoke/verified.json') is None: raise ValueError('Run transfer smoke first')
        print('SMOKE GATE VERIFIED', flush=True)
