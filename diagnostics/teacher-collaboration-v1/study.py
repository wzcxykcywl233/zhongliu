"""Small fixed mechanism study, immutable provenance, validation-only selection."""
import argparse
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
import statistics

import protocol as P
from shared import APP, CHECKOUT, legacy as L, legacy_protocol, load

# Public functions from our own previous package, bound to this protocol module.
# The dependency's source hashes are frozen with this study before execution.
B = load('collaboration_queue_helpers', CHECKOUT / 'diagnostics/teacher-combination-search-v4/study.py')
torch = L.torch
frozen, checked_hashes = L.frozen, L.checked_hashes
make_plan, get_plan, verify_arm = B.make_plan, B.get_plan, B.verify_arm
checked_training, checked_report = B.checked_training, B.checked_report
ARTIFACTS = B.ARTIFACTS


def prepare(root, prior, package, weights, datasets):
    prior_config = P.read(prior / 'frozen-study.json')
    source = {r['File']: r['SHA256'].lower() for r in prior_config['Source']}
    checked_hashes(CHECKOUT, source)
    checked_hashes(weights, {r['Name']: r['SHA256'] for r in prior_config['Weights']})
    cases = L.helpers.verify_datasets(prior_config, datasets)
    images = {k: P.read(prior / 'local-runtime-images.json')[k] for k in ('Training', 'Inference', 'Evaluation')}
    if any(len(v) != 71 or not v.startswith('sha256:') or any(c not in '0123456789abcdef' for c in v[7:]) for v in images.values()):
        raise ValueError('Pinned image IDs required')
    refs = {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']}
    if P.read(prior / 'reference-images.json') != refs:
        raise ValueError('Image identities disagree')
    cfg = asdict(L.get_experiment_config(P.PROFILE)[1])
    if cfg['border_points'] != 1000 or cfg['n_iterations'] != 2:
        raise ValueError('Student inference changed')
    deps = [p for folder in ('frozen-sidecar-teachers-v1', 'frozen-teacher-full-v1', 'teacher-combination-search-v4')
            for p in (CHECKOUT / 'diagnostics' / folder).glob('*.py')]
    manifest = {'schema': 1, 'source_sha256': source, 'prior_sha256': P.sha(prior / 'frozen-study.json'),
                'weights_sha256': {r['Name']: r['SHA256'].lower() for r in prior_config['Weights']},
                'package_sha256': {p.name: P.sha(p) for p in package.iterdir() if p.suffix in ('.py', '.ps1', '.txt')},
                'dependency_sha256': {p.relative_to(CHECKOUT).as_posix(): P.sha(p) for p in deps},
                'trainer_sha256': P.sha(APP / 'ext/co-tracker/train_on_real_data.py'),
                'images': images, 'case_ids': cases, 'pools': P.matrix(), 'steps': P.STEPS,
                'seeds': list(P.SEEDS), 'base_seed': P.BASE_SEED, 'selection_rule': P.RULE,
                'collaboration_rule': P.COLLABORATION_RULE, 'student_config': cfg, 'profile': P.PROFILE,
                'runtime': P.RUNTIME, 'parameters': 28763025, 'trainable_parameters': 3377325,
                'note': 'All teachers frozen, all teacher settings fixed at on3_s0 + scaled control. '
                        'Train only Mamba. One student forward. Only visible coordinate supervision changes. '
                        'Teacher agreement is not ground truth. Public test-38 repeatedly inspected, not blind.'}
    if manifest['trainer_sha256'] != source['cotracker-algorithm/ext/co-tracker/train_on_real_data.py']:
        raise ValueError('Runtime trainer mismatch')
    frozen(root / 'manifest.json', manifest)
    frozen(root / 'runtime.json', P.RUNTIME)
    frozen(root / 'reference-images.json', refs)
    L.freeze(weights / 'scaled_offline.pth', root / 'pretrained-frozen.pth')
    state = L.state_from_checkpoint(torch.load(weights / 'scaled_offline.pth', map_location='cpu', weights_only=True))
    frozen(root / 'reference-proof.json', {'base_tensor_sha256': L.tensor_hash(state), 'checkpoint_sha256': P.sha(root / 'pretrained-frozen.pth')})
    make_plan(root, 'smoke', 'smoke', P.matrix(), [0])
    make_plan(root, 'pilot', 'pilot', {a: P.matrix()[a] for a in P.ARMS}, [0])
    print('PLAN VERIFIED: 8 smoke arms, 6 fixed pilot/full arms; frozen Mamba and teacher pool; 40/10/38 disjoint', flush=True)


def full_tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit(root, stage):
    plan = get_plan(root, stage)
    artifacts, proofs, timing = {}, [], []
    for seed in plan['seeds']:
        reference, initial = verify_arm(root, plan, seed, 'control')
        reference_state = None
        all_rows = {}
        for arm, pool in plan['pools'].items():
            rows, init = (reference, initial) if arm == 'control' else verify_arm(root, plan, seed, arm)
            all_rows[arm] = rows
            if init != initial:
                raise ValueError('Student initialization differs')
            changed = P.paired(reference, rows, plan['pools']['control'], pool)
            folder = root / plan['phase'] / 'train' / f'seed_{seed}' / arm
            exact = None
            if arm in P.NOOPS:
                if reference_state is None:
                    path = root / plan['phase'] / 'train' / f'seed_{seed}/control/cotracker_three_final.pth'
                    reference_state = full_tensor_hash(torch.load(path, map_location='cpu', weights_only=True)['model'])
                exact = full_tensor_hash(torch.load(folder / 'cotracker_three_final.pth', map_location='cpu', weights_only=True)['model']) == reference_state
                if not exact or any(changed) or any(reference[i]['loss'] != r['loss'] for i, r in rows.items()):
                    raise ValueError('Strict no-op student/gradient-path control failed: ' + arm)
            proofs.append({'seed': seed, 'arm': arm, 'steps': len(rows), 'frozen_backbone_exact': True,
                           'noop_full_student_exact': exact, 'changed_supervision_steps_by_slot': changed,
                           'median_steady_step_seconds': statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2)})
            for name in ARTIFACTS:
                artifacts[(folder / name).relative_to(root).as_posix()] = P.sha(folder / name)
            for slot in range(3):
                subset = [r for r in rows.values() if r['primary_index'] == slot]
                if not subset:
                    raise ValueError('Primary slot never exercised')
                c = [r['label_audit']['collaboration'] for r in subset]
                total = lambda key: sum(x[key] for x in c)
                mass = total('base_weight_mass')
                timing.append({'Stage': stage, 'Seed': seed, 'Arm': arm, 'PrimarySlot': slot,
                    'Steps': len(subset), 'StepSeconds': statistics.mean(r['step_seconds'] for r in subset),
                    'TeacherSeconds': statistics.mean(r['label_audit']['teacher_seconds'] for r in subset),
                    'TeacherCalls': statistics.mean(r['label_audit']['teacher_calls'] for r in subset),
                    'Active': total('active'), 'ReviewEligible': total('eligible'), 'Conflict': total('conflict'),
                    'Downweighted': total('downweighted'), 'Fused': total('fused'), 'MissingReview': total('missing_review'),
                    'ConflictFractionEligible': total('conflict') / max(1, total('eligible')),
                    'RetainedWeightMassFraction': total('weight_mass') / mass if mass else 0.,
                    'MaxCorrectionPixels': max(x['max_correction'] for x in c),
                    'MeanDisagreementPixels': total('disagreement_sum') / max(1, total('disagreement_count'))})
        # Reviewer labels and coverage must be paired across all collaborative arms.
        reviewed = [a for a in plan['pools'] if a not in ('control', 'same_teacher')]
        for arm in reviewed[1:]:
            for i, row in all_rows[arm].items():
                q = row['label_audit']['collaboration']
                ref = all_rows[reviewed[0]][i]['label_audit']['collaboration']
                fields = ('slot', *P.LABEL_KEYS, 'support_draw_sha256', 'teacher_queries_sha256')
                if [[x[k] for k in fields] for x in q['reviewers']] != [[x[k] for k in fields] for x in ref['reviewers']]:
                    raise ValueError('Reviewer labels differ across arms')
        if 'gate2' in all_rows and 'shuffle2' in all_rows:
            for i in all_rows['gate2']:
                a, b = [all_rows[k][i]['label_audit']['collaboration'] for k in ('gate2', 'shuffle2')]
                if any(a[k] != b[k] for k in ('eligible', 'conflict', 'downweighted', 'weight_mass', 'base_weight_mass')):
                    raise ValueError('Shuffled review coverage/mass not matched')
    proof = {'complete': True, 'stage': stage, 'plan_sha256': P.sha(root / 'plans' / (stage + '.json')),
             'proofs': proofs, 'artifacts_sha256': artifacts}
    frozen(root / 'audits' / (stage + '.json'), proof)
    (root / 'reports' / stage).mkdir(parents=True, exist_ok=True)
    L.write_csv(root / 'reports' / stage / 'label-runtime.csv', timing)
    frozen(root / 'reports' / stage / 'label-runtime-audit.json', {'sha256': P.sha(root / 'reports' / stage / 'label-runtime.csv'),
           'note': 'Label validity, disagreement and coverage are not correspondence accuracy. No masks used to gate teacher labels.'})
    if stage == 'smoke':
        times = [p['median_steady_step_seconds'] for p in proofs if p['arm'] in P.ARMS]
        L.save_json(root / 'budget-estimate.json', {'smoke_steps': 80, 'pilot_steps': 1500, 'full_steps': 18000,
            'training_only_hours_rough_range': [19500 * min(times) / 3600, 19500 * max(times) / 3600],
            'note': 'Smoke extrapolation only, NOT a bound. Excludes evaluation, loading, saves and audits.'})
    print('AUDIT PASSED ' + stage, flush=True)
    return proof


def select(root, stage):
    scores, parents = checked_report(root, stage)
    plan = get_plan(root, stage)
    if set(plan['pools']) != set(P.ARMS):
        raise ValueError('Fixed mechanism matrix incomplete')
    if stage == 'pilot':
        decision = {'split': 'validation-10', 'full_arms': list(P.ARMS),
                    'rule': 'No pilot elimination: retain paired controls and all fixed mechanisms', 'parents': parents}
        frozen(root / 'confirmation-selection.json', decision)
        make_plan(root, 'confirm', 'full', plan['pools'], list(P.SEEDS), parents,
                  {'confirmation-selection.json': P.sha(root / 'confirmation-selection.json')})
    elif stage == 'confirm':
        candidates = ['tutor010', 'gate2', 'gate4', 'fuse2']  # shuffle2 is a mechanism control, not an adoption candidate.
        selected = P.best(scores, candidates)
        decision = {'split': 'validation-10', 'selected_overall': selected, 'test_arms': list(P.ARMS),
                    'rule': P.RULE, 'parents': parents,
                    'evidence': {a: P.selection_evidence(scores[a], scores['control']) for a in candidates},
                    'note': 'All fixed mechanisms reported, including failures. No test selection. '
                            'Teacher coordination success requires checking gate2 against shuffle2 and tutor010.'}
        frozen(root / 'selection.json', decision)
        make_plan(root, 'test', 'full', plan['pools'], list(P.SEEDS), parents,
                  {'selection.json': P.sha(root / 'selection.json')})
    else:
        raise ValueError('Invalid selection stage')
    print('SELECTION FROZEN after ' + stage, flush=True)


def official_report(root, stage, split):
    if split == 'test-38' and stage != 'test':
        raise ValueError('Only frozen final selection may access test-38')
    plan = get_plan(root, stage)
    training = checked_training(root, stage)
    manifest = P.read(root / 'manifest.json')
    expected_cases = set(manifest['case_ids'][split])
    location = root / plan['phase'] / split
    target = root / 'reports' / stage / split
    if (target / 'audit.json').exists():
        checked_hashes(root, P.read(target / 'audit.json')['artifacts_sha256'])
    records, scores, pairs, case_pairs, hashes = [], {}, [], [], dict(training['artifacts_sha256'])

    def load(relative, kind):
        folder = location / relative
        seed, arm = (-1, 'pretrained') if relative == 'pretrained' else (int(relative.split('/')[0][5:]), relative.split('/')[1])
        if P.read(folder / 'evaluation-inputs.json') != B.evaluation_inputs(root, plan, seed, arm):
            raise ValueError('Evaluated checkpoint binding changed')
        hashes[(folder / 'evaluation-inputs.json').relative_to(root).as_posix()] = P.sha(folder / 'evaluation-inputs.json')
        if P.read(folder / 'frozen-images.json') != P.read(root / 'reference-images.json'):
            raise ValueError('Evaluation image identity changed')
        values, cases, _, _ = L.load_arm(folder, kind, len(expected_cases), 28763025 if kind == 'mamba' else 25385700)
        if set(cases) != expected_cases:
            raise ValueError('Wrong official case IDs')
        metric = folder / P.PROFILE / 'metrics.json'
        hashes[metric.relative_to(root).as_posix()] = P.sha(metric)
        for case in cases:
            job = folder / P.PROFILE / 'checkpoint/jobs' / case
            diag = P.read(job / 'diagnostics.json')
            if diag['config'] != manifest['student_config'] or (kind == 'mamba' and diag['model']['sidecar_runtime'] != P.RUNTIME):
                raise ValueError('Fixed student inference changed')
            for name in ('diagnostics.json', '.complete', 'output/images/mri-linac-series-targets/output.mha'):
                path = job / name
                hashes[path.relative_to(root).as_posix()] = P.sha(path)
        return values, cases

    pretrained, pretrained_cases = load('pretrained', 'none')
    scores['pretrained'] = [pretrained]
    records.append({'Split': split, 'Seed': -1, 'Arm': 'pretrained', **pretrained})
    for seed in plan['seeds']:
        loaded = {a: load(f'seed_{seed}/{a}', 'mamba') for a in plan['pools']}
        for arm, (values, cases) in loaded.items():
            scores.setdefault(arm, []).append(values)
            records.append({'Split': split, 'Seed': seed, 'Arm': arm, **values})
            references = ['control', 'pretrained']
            for reference in dict.fromkeys(references):
                if arm == reference:
                    continue
                ref, ref_cases = (pretrained, pretrained_cases) if reference == 'pretrained' else loaded[reference]
                pairs.append({'Split': split, 'Seed': seed, 'Candidate': arm, 'Reference': reference,
                              **{'Delta_' + k: values[k] - ref[k] for k in P.FIELDS}})
                for case in sorted(cases):
                    case_pairs.append({'Split': split, 'Seed': seed, 'Candidate': arm, 'Reference': reference, 'Case': case,
                                       **{'Delta_' + k: float(cases[case][f]) - float(ref_cases[case][f]) for k, f in L.FIELDS.items()}})
    summary = []
    selection = P.read(root / 'selection.json') if stage == 'test' else None
    for arm, samples in scores.items():
        row = {'Split': split, 'Arm': arm, 'N': len(samples), 'StudentProfile': P.PROFILE,
               'Parameters': 25385700 if arm == 'pretrained' else 28763025}
        if selection is not None:
            roles = []
            if arm == 'pretrained': roles.append('official_no_sidecar')
            if arm == 'control': roles.append('current_teacher_pool')
            if arm == selection['selected_overall']: roles.append('validation_selected_overall')
            if arm == 'shuffle2': roles.append('coverage_matched_negative_control')
            if arm == 'tutor010': roles.append('coordinate_only_tutor_control')
            row['SelectionRole'] = ';'.join(roles) or 'matched_ablation_diagnostic'
        for k in (*P.FIELDS, 'TimeSec'):
            row[k + '_Mean'] = statistics.mean(r[k] for r in samples)
            row[k + '_SD'] = statistics.stdev(r[k] for r in samples) if len(samples) > 1 else 0.
        summary.append(row)
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in (('summary', summary), ('results', records), ('paired-deltas', pairs), ('paired-cases', case_pairs)):
        L.write_csv(target / (name + '.csv'), rows)
    score_path = target / 'scores.json'
    L.save_json(score_path, scores)
    hashes[score_path.relative_to(root).as_posix()] = P.sha(score_path)
    for name in ('summary', 'results', 'paired-deltas', 'paired-cases'):
        path = target / (name + '.csv')
        hashes[path.relative_to(root).as_posix()] = P.sha(path)
    frozen(target / 'audit.json', {'complete': True, 'stage': stage, 'split': split, 'cases': sorted(expected_cases),
                'plan_sha256': P.sha(root / 'plans' / (stage + '.json')), 'artifacts_sha256': hashes,
                'note': 'Official metrics copied, not recomputed. N denotes training seeds. Validation used repeatedly; '
                        'test-38 already inspected historically, no blind-test or p-value claim.'})
    print('REPORT COMPLETE ' + stage + '/' + split, flush=True)
    for r in summary:
        print(f"RESULT {r['Arm']}: DSC={r['DSC_Mean']:.6f} +/- {r['DSC_SD']:.6f}; D98={r['D98_Mean']:.6f}", flush=True)


def report(root, stage, split):
    official_report(root, stage, split)
    target = root / 'reports' / stage / split
    scores = P.read(target / 'scores.json')
    plan = get_plan(root, stage)
    comparisons = [('gate2', 'shuffle2'), ('gate2', 'tutor010'), ('gate4', 'gate2'), ('fuse2', 'gate2')]
    rows = [{'Split': split, 'Seed': seed, 'Candidate': a, 'Reference': b,
             **{'Delta_' + k: scores[a][i][k] - scores[b][i][k] for k in P.FIELDS}}
            for i, seed in enumerate(plan['seeds']) for a, b in comparisons]
    L.write_csv(target / 'mechanism-deltas.csv', rows)
    with (target / 'paired-cases.csv').open(encoding='utf-8-sig', newline='') as stream:
        case_deltas = {(int(r['Seed']), r['Candidate'], r['Case']): r for r in csv.DictReader(stream)
                       if r['Reference'] == 'control'}
    cases = P.read(root / 'manifest.json')['case_ids'][split]
    case_rows = [{'Split': split, 'Seed': seed, 'Candidate': a, 'Reference': b, 'Case': case,
                 **{'Delta_' + k: float(case_deltas[seed, a, case]['Delta_' + k]) -
                                   float(case_deltas[seed, b, case]['Delta_' + k]) for k in P.FIELDS}}
                for seed in plan['seeds'] for a, b in comparisons for case in cases]
    L.write_csv(target / 'mechanism-cases.csv', case_rows)
    frozen(target / 'mechanism-audit.json', {'source_audit_sha256': P.sha(target / 'audit.json'),
           'mechanism_deltas_sha256': P.sha(target / 'mechanism-deltas.csv'),
           'mechanism_cases_sha256': P.sha(target / 'mechanism-cases.csv'),
           'note': 'Coordinate-only tutor, not exact historical two-student-forward tutor. '
                   'Seed SD is not patient uncertainty. Coverage-matched shuffle distinguishes informative review from reweighting.'})


def teacher_smoke(root, weights):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.cotracker.losses import sequence_loss
    torch.manual_seed(P.BASE_SEED)
    models = [(P.TEACHER_TYPES[s], build_cotracker(str(weights / cfg['checkpoint']), offline=s == 2,
                window_len=60 if s == 2 else 16, v2=s == 0).cuda().eval().requires_grad_(False))
              for s, cfg in enumerate(P.B.baseline())]
    before = [full_tensor_hash(m.state_dict()) for _, m in models]
    video = torch.rand(1, 60, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [29., 32., 32.], [59., 48., 48.]]], device='cuda')
    proofs = []
    for slot in range(3):
        control = None
        for arm, pool in P.matrix().items():
            coordinator = P.Coordinator(arm, pool)
            p, v, c, audit = coordinator.dispatch(slot, *models[slot], video, queries,
                                                 P.support_for(slot, video, queries), arm, models)
            pred = (p + 1.).detach().requires_grad_()
            value = coordinator.loss(sequence_loss, [[pred]], [p], [torch.ones_like(v)],
                                     vis=[(v > .9).float()], add_huber_loss=True, loss_only_for_visible=True)
            value.backward()
            if not bool(torch.isfinite(value)) or not bool(torch.isfinite(pred.grad).all()):
                raise ValueError('Nonfinite GPU collaboration loss/gradient')
            if arm == 'control':
                control = p.clone(), value.detach().clone(), pred.grad.clone()
            if arm in P.NOOPS and not all(torch.equal(x, y) for x, y in zip(control, (p, value.detach(), pred.grad))):
                raise ValueError('GPU no-op labels/loss/gradient differ')
            proofs.append({'slot': slot, 'arm': arm, 'audit': audit})
    if before != [full_tensor_hash(m.state_dict()) for _, m in models]:
        raise ValueError('Teacher weights changed')
    L.save_json(root / 'teacher-interface-smoke.json', {'complete': True, 'manifest_sha256': P.sha(root / 'manifest.json'),
                'profiles': proofs, 'no_op_baselines_exact': True, 'teachers_frozen_exact': True,
                'query_times': [0, 29, 59], 'note': 'Real teacher interfaces plus loss gradients; synthetic unit tests exercise conflict/fusion triggers.'})
    print('GPU TEACHER COLLABORATION PASSED 8 arms x 3 primary slots', flush=True)


def smoke_gate(root):
    proof = checked_training(root, 'smoke')
    gate = P.read(root / 'smoke-gate.json')
    interface = P.read(root / 'teacher-interface-smoke.json')
    if (not gate['complete'] or gate['manifest_sha256'] != P.sha(root / 'manifest.json') or
        gate['audit_sha256'] != P.sha(root / 'audits/smoke.json') or
        gate['interface_sha256'] != P.sha(root / 'teacher-interface-smoke.json') or
        not interface['complete'] or not interface['no_op_baselines_exact'] or not interface['teachers_frozen_exact'] or
        len(interface['profiles']) != 24 or interface['manifest_sha256'] != P.sha(root / 'manifest.json') or
        {p['arm'] for p in proof['proofs']} != set(P.matrix()) or
        any(p['noop_full_student_exact'] is not True for p in proof['proofs'] if p['arm'] in P.NOOPS)):
        raise ValueError('Collaboration smoke gate incomplete')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'runtime', 'verify-arm', 'audit', 'report', 'select', 'teacher-smoke',
                   'gpu-smoke', 'smoke-gate', 'verify-training', 'plan-check', 'eval-bind'))
    p.add_argument('--root', type=Path, default=Path('/study'))
    p.add_argument('--prior', type=Path, default=Path('/prior'))
    p.add_argument('--package', type=Path, default=Path('/package'))
    p.add_argument('--weights', type=Path, default=Path('/checkpoints'))
    p.add_argument('--datasets', type=Path, default=Path('/datasets'))
    p.add_argument('--stage', choices=('smoke', 'pilot', 'confirm', 'test'), default='smoke')
    p.add_argument('--split', choices=('validation-10', 'test-38'), default='validation-10')
    p.add_argument('--kind', choices=('Training', 'Inference'), default='Training')
    p.add_argument('--seed', type=int, choices=(-1, *P.SEEDS), default=0)
    p.add_argument('--arm', default='control')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.command != 'prepare':
        manifest = P.read(a.root / 'manifest.json')
        checked_hashes(CHECKOUT, manifest['dependency_sha256'])
        checked_hashes(a.package, manifest['package_sha256'])
    if a.command == 'prepare': prepare(a.root, a.prior, a.package, a.weights, a.datasets)
    elif a.command == 'runtime': L.runtime(a.root, a.kind)
    elif a.command == 'verify-arm': verify_arm(a.root, get_plan(a.root, a.stage), a.seed, a.arm)
    elif a.command == 'audit': audit(a.root, a.stage)
    elif a.command == 'report': report(a.root, a.stage, a.split)
    elif a.command == 'select': select(a.root, a.stage)
    elif a.command == 'teacher-smoke': teacher_smoke(a.root, a.weights)
    elif a.command == 'gpu-smoke': B.gpu_smoke(a.root)
    elif a.command == 'smoke-gate': smoke_gate(a.root)
    elif a.command == 'verify-training': checked_training(a.root, a.stage)
    elif a.command == 'eval-bind': B.bind_evaluation(a.root, a.stage, a.split, a.seed, a.arm)
    else: get_plan(a.root, a.stage, deep=True)


if __name__ == '__main__':
    main()
