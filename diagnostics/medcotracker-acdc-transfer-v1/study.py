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
    if phase == 'pilot' and marker(root, 'smoke/verified.json') is None:
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
            if marker(root, f'pilot/external/{arm}/verified.json') is None:
                raise ValueError('External pilot incomplete')
            source = root / 'pilot/external' / arm / 'external-final.pth'
            expected_steps = 500
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
    fresh = [marker(root, f'{phase}/init/{a}.json')['sidecar_hashes'] for a in ('R1', 'B1', 'B2', 'B3')]
    if any(value != fresh[0] for value in fresh):
        raise ValueError('Fresh sidecar initializations not paired')
    if phase == 'pilot':
        signatures = []
        for arm in S.ROUTES:
            proof = marker(root, f'pilot/external/{arm}/verified.json')
            if proof is None or proof['steps'] != 500:
                raise ValueError('External stage incomplete')
            records = read(root / 'pilot/external' / arm / 'steps.json')
            if [r['step'] for r in records] != list(range(500)):
                raise ValueError('Missing external steps')
            signatures.append([(r['patient'], r['slot'], r['frames'], r['label_sha256']) for r in records])
        if any(x != signatures[0] for x in signatures):
            raise ValueError('External clips or labels not paired')
        for step, (_, _, _, expected) in enumerate(signatures[0]):
            if digest(root / 'pilot/labels' / f'{step:04}.pth') != expected:
                raise ValueError('External label cache changed')
    files = [root / phase / 'train' / arm / name for arm in ARMS
             for name in ('verified.json', 'cotracker_three_final.pth', 'paired-steps.jsonl')]
    files += [root / phase / 'init' / (arm + suffix) for arm in ARMS for suffix in ('.json', '.pth')]
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
    publish(root, 'smoke/verified.json', files, stage='TrackRAD-transfer-smoke', steps_per_arm=10)
    print('GPU TRANSFER SMOKE PASSED all five arms', flush=True)


def eval_bind(root, arm):
    plan = checked_plan(root)
    if marker(root, 'pilot/training-audit.json') is None:
        raise ValueError('Pilot audit missing')
    checkpoint = root / ('pretrained-frozen.pth' if arm == 'pretrained' else f'pilot/train/{arm}/cotracker_three_final.pth')
    value = {'manifest_sha256': digest(root / 'manifest.json'), 'checkpoint_sha256': digest(checkpoint),
             'profile': PROFILE, 'runtime': RUNTIME, 'split': 'validation-10', 'images': plan['images'],
             'case_ids': plan['trackrad_ids']['validation-10']}
    S.frozen(root / 'pilot/validation-10' / arm / 'evaluation-inputs.json', value)


def report(root):
    plan = checked_plan(root)
    audit(root, 'pilot')
    from summarize_feature_sidecar import load_arm
    from summarize_backbone_growth import FIELDS, write_csv
    rows, cases, files = [], {}, []
    for arm in ('pretrained',) + ARMS:
        eval_bind(root, arm)
        folder = root / 'pilot/validation-10' / arm
        if read(folder / 'frozen-images.json') != read(root / 'reference-images.json'):
            raise ValueError('Evaluation images changed')
        values, case_rows, _, rms = load_arm(folder, 'none' if arm == 'pretrained' else 'mamba', 10,
                                             25385700 if arm == 'pretrained' else 28763025)
        if set(case_rows) != set(plan['trackrad_ids']['validation-10']):
            raise ValueError('Wrong validation cases')
        for case in case_rows:
            job = folder / PROFILE / 'checkpoint/jobs' / case
            d = read(job / 'diagnostics.json')
            if d['config'] != plan['student_config']:
                raise ValueError('Frozen inference configuration changed')
            if arm != 'pretrained' and d['model']['sidecar_runtime'] != RUNTIME:
                raise ValueError('Runtime controls changed during validation')
            files.extend(job / name for name in ('diagnostics.json', '.complete', 'output/images/mri-linac-series-targets/output.mha'))
        cases[arm] = case_rows
        rows.append({'Split': 'validation-10', 'Arm': arm, 'N': 1,
                     'ExternalSteps': 0 if arm in ('pretrained', 'R1') else 500,
                     'TrackRADSteps': 0 if arm == 'pretrained' else 250, 'ResidualRmsMax': rms, **values})
        files += [folder / PROFILE / 'metrics.json', folder / 'evaluation-inputs.json']
    ref = next(r for r in rows if r['Arm'] == 'R1')
    deltas = [{'Candidate': r['Arm'], 'Reference': 'R1', **{'Delta_' + k: r[k] - ref[k] for k in FIELDS}}
              for r in rows if r['Arm'] != 'R1']
    paired = [{'Candidate': arm, 'Reference': 'R1', 'Case': case,
               **{'Delta_' + k: float(cases[arm][case][f]) - float(cases['R1'][case][f]) for k, f in FIELDS.items()}}
              for arm in S.ROUTES for case in sorted(cases['R1'])]
    dest = root / 'reports/validation-10'
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / 'summary.csv', rows)
    write_csv(dest / 'deltas-vs-R1.csv', deltas)
    write_csv(dest / 'paired-cases.csv', paired)
    eligible = selection(rows)
    save(dest / 'selection.json', {'eligible_routes': eligible, 'suggested_confirmation': eligible[:1] or ['R1'],
                                  'rule': RULE['selection'], 'no_automatic_full_training': True,
                                  'note': 'Single-seed pilot candidates, not proven generalization or tracking ground truth.'})
    files += [dest / name for name in ('summary.csv', 'deltas-vs-R1.csv', 'paired-cases.csv', 'selection.json')]
    publish(root, 'summary.json', files, stage='single-seed-pilot', split='validation-10',
            external_steps=500, transfer_steps=250, test_access=False, formal_training_started=False)
    for row in rows:
        print(f'RESULT {row["Arm"]}: DSC={row["DSC"]:.6f}; D98={row["D98"]:.6f}', flush=True)
    print('PILOT COMPLETE validation-10 only; no automatic next stage', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('plan', 'handoff', 'audit', 'runtime', 'inference-smoke', 'gate', 'eval-bind', 'report'))
    p.add_argument('--phase', choices=('smoke', 'pilot'), default='smoke')
    p.add_argument('--arm', choices=('pretrained',) + ARMS, default='R1')
    a = p.parse_args()
    root = Path('/study')
    if a.command == 'plan':
        prepare(root, Path('/prior'), Path('/audit'), Path('/external-smoke'), Path('/weights'),
                Path('/datasets/train'), Path('/datasets/validation'), read(Path('/prior/local-runtime-images.json')))
    elif a.command == 'handoff': handoff(root, a.phase, a.arm, Path('/weights'), Path('/external-smoke'))
    elif a.command == 'audit': audit(root, a.phase)
    elif a.command == 'runtime': runtime(root)
    elif a.command == 'inference-smoke': inference_smoke(root)
    elif a.command == 'eval-bind': eval_bind(root, a.arm)
    elif a.command == 'report': report(root)
    else:
        checked_plan(root)
        if marker(root, 'smoke/verified.json') is None: raise ValueError('Run transfer smoke first')
        print('SMOKE GATE VERIFIED', flush=True)
