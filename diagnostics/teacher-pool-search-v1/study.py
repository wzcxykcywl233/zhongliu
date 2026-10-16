"""Immutable staged search, frozen-student audits and automatic final test report."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
from pathlib import Path
import os
import statistics

import protocol as P
from shared import APP, CHECKOUT, legacy as L, legacy_protocol

torch = L.torch
frozen, checked_hashes = L.frozen, L.checked_hashes
ARTIFACTS = ('cotracker_three_final.pth', 'run-config.json', 'initialization-report.json', 'trainer-adapter.json', 'paired-steps.jsonl')


def prepare(root, prior, package, weights, datasets):
    prior_config = P.read(prior / 'frozen-study.json')
    source = {r['File']: r['SHA256'].lower() for r in prior_config['Source']}
    checked_hashes(CHECKOUT, source)
    checked_hashes(weights, {r['Name']: r['SHA256'] for r in prior_config['Weights']})
    cases = L.helpers.verify_datasets(prior_config, datasets)
    images = P.read(prior / 'local-runtime-images.json')
    images = {k: images[k] for k in ('Training', 'Inference', 'Evaluation')}
    if any(len(v) != 71 or not v.startswith('sha256:') or any(c not in '0123456789abcdef' for c in v[7:]) for v in images.values()):
        raise ValueError('Pinned local image IDs required')
    refs = {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']}
    if P.read(prior / 'reference-images.json') != refs:
        raise ValueError('Image IDs disagree')
    pools, families = P.matrix()
    cfg = asdict(L.get_experiment_config(P.PROFILE)[1])
    if cfg['border_points'] != 1000 or cfg['n_iterations'] != 2:
        raise ValueError('Fixed student inference changed')
    dependencies = [p for folder in ('frozen-sidecar-teachers-v1', 'frozen-teacher-full-v1')
                    for p in (CHECKOUT / 'diagnostics' / folder).glob('*.py')]
    manifest = {'schema': 1, 'source_sha256': source, 'prior_sha256': P.sha(prior / 'frozen-study.json'),
                'weights_sha256': {r['Name']: r['SHA256'].lower() for r in prior_config['Weights']},
                'package_sha256': {p.name: P.sha(p) for p in package.iterdir() if p.suffix in ('.py', '.ps1', '.txt')},
                'dependency_sha256': {p.relative_to(CHECKOUT).as_posix(): P.sha(p) for p in dependencies},
                'trainer_sha256': P.sha(APP / 'ext/co-tracker/train_on_real_data.py'),
                'images': images, 'case_ids': cases, 'pools': pools, 'families': families, 'selection_rule': P.RULE,
                'steps': P.STEPS, 'seeds': list(P.SEEDS), 'base_seed': P.BASE_SEED,
                'student_config': cfg, 'profile': P.PROFILE, 'runtime': P.RUNTIME,
                'parameters': 28763025, 'trainable_parameters': 3377325,
                'note': 'Freeze official backbone; train fixed Mamba only. Candidate screening uses STUDENT validation-10, '
                        'not teacher confidence. No teacher fine-tuning. Public test-38 is repeatedly inspected, not blind.'}
    if manifest['trainer_sha256'] != source['cotracker-algorithm/ext/co-tracker/train_on_real_data.py']:
        raise ValueError('Runtime trainer differs from frozen checkout')
    frozen(root / 'manifest.json', manifest)
    frozen(root / 'runtime.json', P.RUNTIME)
    frozen(root / 'reference-images.json', refs)
    L.freeze(weights / 'scaled_offline.pth', root / 'pretrained-frozen.pth')
    state = L.state_from_checkpoint(torch.load(weights / 'scaled_offline.pth', map_location='cpu', weights_only=True))
    frozen(root / 'reference-proof.json', {'base_tensor_sha256': L.tensor_hash(state), 'checkpoint_sha256': P.sha(root / 'pretrained-frozen.pth')})
    make_plan(root, 'smoke', 'smoke', {p: pools[p] for p in ('control', 'v2_s0', 'on3_s0', 'offold_i6', 'offscaled_slo')}, [0])
    make_plan(root, 'pilot', 'pilot', pools, [0])
    print('PLAN VERIFIED: 4 teacher families, 22 unique pilot pools; frozen Mamba, 40/10/38 disjoint', flush=True)


def make_plan(root, stage, phase, pools, seeds, parents=None, decisions=None):
    plan = {'stage': stage, 'phase': phase, 'pools': pools, 'seeds': seeds,
            'steps': P.STEPS[phase], 'manifest_sha256': P.sha(root / 'manifest.json'),
            'parents': parents or {}, 'decisions': decisions or {}}
    frozen(root / 'plans' / (stage + '.json'), plan)
    for seed, arm in itertools.product(seeds, pools):
        frozen(root / phase / 'train' / f'seed_{seed}' / arm / 'run-config.json',
               P.config(seed, arm, phase, pools[arm], plan['manifest_sha256']))
    return plan


def get_plan(root, stage, deep=False):
    plan = P.read(root / 'plans' / (stage + '.json'))
    if plan['stage'] != stage or plan['manifest_sha256'] != P.sha(root / 'manifest.json') or 'control' not in plan['pools']:
        raise ValueError('Plan changed')
    if plan['steps'] != P.STEPS[plan['phase']] or plan['seeds'] != ([0] if plan['phase'] != 'full' else list(P.SEEDS)):
        raise ValueError('Wrong phase budget')
    checked_hashes(root, plan['parents'])
    checked_hashes(root, plan['decisions'])
    if deep:
        for relative in plan['parents']:
            proof = P.read(root / relative)
            checked_hashes(root, proof['artifacts_sha256'])
    return plan


def verify_arm(root, plan, seed, arm):
    folder = root / plan['phase'] / 'train' / f'seed_{seed}' / arm
    cfg = P.config(seed, arm, plan['phase'], plan['pools'][arm], P.sha(root / 'manifest.json'))
    if P.read(folder / 'run-config.json') != cfg:
        raise ValueError('Arm configuration changed')
    init = P.read(folder / 'initialization-report.json')
    info = init['sidecar']
    base_hash = P.read(root / 'reference-proof.json')['base_tensor_sha256']
    if (init['architecture'] != 'base' or not init['initialization_outputs_exact'] or info['kind'] != 'mamba' or
        not info['zero_projection_verified'] or not info['initialization_outputs_exact'] or any(info['initialization_max_abs_difference']) or
        info['initialization_reference'] != 'frozen-pretrained' or info['base_tensor_sha256'] != base_hash or
        info['execution_runtime'] != P.RUNTIME or info['parameters'] != 28763025 or info['trainable_parameters'] != 3377325):
        raise ValueError('Frozen backbone/initialization proof failed')
    manifest = P.read(root / 'manifest.json')
    source = P.instrument((APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8'), legacy_protocol.instrument)
    expected = {'source_sha256': manifest['trainer_sha256'], 'adapter_sha256': manifest['package_sha256']['train.py'],
                'protocol_sha256': manifest['package_sha256']['protocol.py'],
                'executed_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'pool_sha256': cfg['pool_sha256'], 'stop_steps': cfg['steps'], 'scheduler_steps': 1000}
    if P.read(folder / 'trainer-adapter.json') != expected:
        raise ValueError('Executed adapter differs')
    final = torch.load(folder / 'cotracker_three_final.pth', map_location='cpu', weights_only=True)
    if (final['total_steps'] != cfg['steps'] or final['feature_sidecar'] != 'mamba' or final['backbone_architecture'] != 'base' or
        final['teacher_student_policy'] != arm or final['teacher_pool_sha256'] != cfg['pool_sha256']):
        raise ValueError('Wrong checkpoint/teacher pool')
    model = L.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    L.attach_sidecar(model, 'mamba')
    model.load_state_dict(final['model'], strict=True)
    if (sum(p.numel() for p in model.parameters()) != 28763025 or
        sum(p.numel() for p in model.parameters() if p.requires_grad) != 3377325 or L.tensor_hash(model.state_dict()) != base_hash or
        any(not bool(torch.isfinite(p).all()) for p in model.parameters()) or
        not bool(model.updateformer.feature_sidecar.output.weight.count_nonzero())):
        raise ValueError('Frozen base/learned branch/parameter budget changed')
    rows = P.records(folder / 'paired-steps.jsonl', cfg)
    sampler = L.TeacherPairSampler(3, seed=P.BASE_SEED + seed)
    if any(rows[i]['primary_index'] != sampler.sample().primary for i in range(cfg['steps'])):
        raise ValueError('Teacher sampling sequence changed')
    return rows, info['initial_sidecar_sha256']


def audit(root, stage):
    plan = get_plan(root, stage)
    artifacts, proofs, timing = {}, [], []
    for seed in plan['seeds']:
        reference, initial = verify_arm(root, plan, seed, 'control')
        for arm, pool in plan['pools'].items():
            rows, init = (reference, initial) if arm == 'control' else verify_arm(root, plan, seed, arm)
            if init != initial:
                raise ValueError('Initial branch differs')
            changed = P.paired(reference, rows, plan['pools']['control'], pool)
            proofs.append({'seed': seed, 'arm': arm, 'steps': len(rows), 'frozen_backbone_exact': True,
                           'changed_label_steps_by_slot': changed})
            folder = root / plan['phase'] / 'train' / f'seed_{seed}' / arm
            for name in ARTIFACTS:
                path = folder / name
                artifacts[path.relative_to(root).as_posix()] = P.sha(path)
            for slot in range(3):
                subset = [r for r in rows.values() if r['primary_index'] == slot]
                if not subset:
                    raise ValueError('Teacher slot was never exercised')
                timing.append({'Stage': stage, 'Seed': seed, 'Arm': arm, 'Slot': slot, 'Steps': len(subset),
                               'TeacherSeconds': statistics.mean(r['label_audit']['teacher_seconds'] for r in subset),
                               'StepSeconds': statistics.mean(r['step_seconds'] for r in subset),
                               **{k: statistics.mean(r['label_audit'][k] for r in subset) for k in
                                  ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction')}})
    proof = {'complete': True, 'stage': stage, 'plan_sha256': P.sha(root / 'plans' / (stage + '.json')),
             'proofs': proofs, 'artifacts_sha256': artifacts}
    frozen(root / 'audits' / (stage + '.json'), proof)
    (root / 'reports' / stage).mkdir(parents=True, exist_ok=True)
    L.write_csv(root / 'reports' / stage / 'label-runtime.csv', timing)
    print('AUDIT PASSED ' + stage, flush=True)
    return proof


def checked_training(root, stage):
    plan = get_plan(root, stage, deep=True)
    proof = P.read(root / 'audits' / (stage + '.json'))
    wanted = {(s, p) for s in plan['seeds'] for p in plan['pools']}
    if (not proof['complete'] or proof['stage'] != stage or proof['plan_sha256'] != P.sha(root / 'plans' / (stage + '.json')) or
        len(proof['proofs']) != len(wanted) or {(r['seed'], r['arm']) for r in proof['proofs']} != wanted or
        any(not r['frozen_backbone_exact'] or r['steps'] != plan['steps'] for r in proof['proofs'])):
        raise ValueError('Training audit incomplete')
    expected = {f"{plan['phase']}/train/seed_{s}/{a}/{n}" for s, a in wanted for n in ARTIFACTS}
    if set(proof['artifacts_sha256']) != expected:
        raise ValueError('Training artifact inventory incomplete')
    checked_hashes(root, proof['artifacts_sha256'])
    return proof


def report(root, stage, split):
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
        if P.read(folder / 'evaluation-inputs.json') != evaluation_inputs(root, plan, seed, arm):
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
            references = ['control', 'oldpool' if 'oldpool' in loaded else 'control', 'pretrained']
            # Multi-slot combinations must also be compared with their component
            # single-slot changes, rather than only the original teacher pool.
            pool = plan['pools'][arm]
            if sum(x != y for x, y in zip(pool, plan['pools']['control'])) >= 2:
                for other, settings in plan['pools'].items():
                    changed_slots = [s for s in range(3) if settings[s] != plan['pools']['control'][s]]
                    if len(changed_slots) == 1 and settings[changed_slots[0]] == pool[changed_slots[0]]:
                        references.append(other)
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
            if arm == 'oldpool': roles.append('old_teacher_pool')
            if arm == selection['selected_overall']: roles.append('validation_selected_overall')
            if selection['qualified_multi_slot_combination'] and arm == selection['best_combination']:
                roles.append('validation_selected_combination')
            row['SelectionRole'] = ';'.join(roles) or 'single_slot_candidate'
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


def checked_report(root, stage):
    folder = root / 'reports' / stage / 'validation-10'
    proof = P.read(folder / 'audit.json')
    plan = get_plan(root, stage)
    if (not proof['complete'] or proof['stage'] != stage or proof['split'] != 'validation-10' or
        proof['cases'] != sorted(P.read(root / 'manifest.json')['case_ids']['validation-10']) or
        proof['plan_sha256'] != P.sha(root / 'plans' / (stage + '.json'))):
        raise ValueError('Validation proof incomplete')
    checked_hashes(root, proof['artifacts_sha256'])
    checked_training(root, stage)
    scores = P.read(folder / 'scores.json')
    if set(scores) != {'pretrained', *plan['pools']} or any(len(scores[p]) != len(plan['seeds']) for p in plan['pools']):
        raise ValueError('Incomplete validation score matrix')
    return scores, {(folder / 'audit.json').relative_to(root).as_posix(): P.sha(folder / 'audit.json')}


def select(root, stage):
    scores, parent = checked_report(root, stage)
    manifest = P.read(root / 'manifest.json')
    if stage == 'pilot':
        winners = {family: P.best(scores, names, names[0]) for family, names in manifest['families'].items()}
        pools = {a: manifest['pools'][a] for a in dict.fromkeys(['control', 'oldpool', *winners.values()])}
        frozen(root / 'pilot-selection.json', {'split': 'validation-10', 'winners': winners, 'rule': P.RULE, 'parents': parent})
        make_plan(root, 'confirm', 'full', pools, list(P.SEEDS), parent,
                  {'pilot-selection.json': P.sha(root / 'pilot-selection.json')})
    elif stage == 'confirm':
        first = P.read(root / 'pilot-selection.json')['winners']
        slot_winners = {0: P.best(scores, [first['v2']]), 1: P.best(scores, [first['on3']]),
                        2: P.best(scores, [first['offold'], first['offscaled']])}
        pools = get_plan(root, 'confirm')['pools']
        combined = P.combinations(pools, slot_winners)
        combined['oldpool'] = pools['oldpool']
        frozen(root / 'slot-selection.json', {'split': 'validation-10', 'winners': {str(k): v for k, v in slot_winners.items()}, 'rule': P.RULE, 'parents': parent})
        make_plan(root, 'combine', 'full', combined, list(P.SEEDS), parent,
                  {'slot-selection.json': P.sha(root / 'slot-selection.json')})
    elif stage == 'combine':
        pools = get_plan(root, 'combine')['pools']
        multi = [a for a, p in pools.items() if sum(x != y for x, y in zip(p, pools['control'])) >= 2]
        combo = P.best(scores, multi)
        selected = P.best(scores, list(pools))
        singles = [a for a, p in pools.items() if sum(x != y for x, y in zip(p, pools['control'])) == 1]
        names = list(dict.fromkeys(['control', 'oldpool', *singles, combo, selected]))
        selection = {'split': 'validation-10', 'best_combination': combo, 'selected_overall': selected,
                     'qualified_multi_slot_combination': combo in multi, 'test_arms': names, 'rule': P.RULE, 'parents': parent,
                     'note': 'Relative winner within this finite staged search; no improvement means keep control. '
                             'Do not reselect after test-38. All single-family and failed-combination validation reports retained.'}
        frozen(root / 'selection.json', selection)
        make_plan(root, 'test', 'full', {a: pools[a] for a in names}, list(P.SEEDS), parent,
                  {'selection.json': P.sha(root / 'selection.json')})
    else:
        raise ValueError('Invalid selection stage')
    print('SELECTION FROZEN after ' + stage, flush=True)


def teacher_smoke(root, weights):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_points_on_a_grid
    torch.manual_seed(P.BASE_SEED)
    video = torch.rand(1, 60, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [29., 32., 32.], [59., 48., 48.]]], device='cuda')
    xy = get_points_on_a_grid(5, (64, 64), device=video.device)
    grid = torch.cat((torch.zeros_like(xy[..., :1]), xy), -1)
    draw = torch.rand(1, 100, 3, device='cuda')
    draw[..., 0] = (draw[..., 0] * 60).floor()
    draw[..., 1:] *= 63
    pools, groups = P.matrix()
    proofs = []
    for family in P.FAMILIES:
        slot = P.FAMILY_SLOT[family]
        setting = pools[groups[family][0]][slot]
        model = build_cotracker(str(weights / setting['checkpoint']), offline=slot == 2, window_len=60 if slot == 2 else 16, v2=slot == 0).cuda().eval()
        model.requires_grad_(False)
        before = L.tensor_hash(model.state_dict())
        support = grid if slot < 2 else draw
        # Compare no-op adapter with original model call, including default iters=4.
        with torch.no_grad():
            original = model(video, torch.cat((queries, support), 1))
        for arm in groups[family]:
            p, v, c, info = P.teacher_forward(slot, ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot],
                                             model, video, queries, support, arm, pools[arm])
            if arm == groups[family][0]:
                expected = (original[0][:, :, :3], original[1][:, :, :3],
                            torch.ones_like(original[1][:, :, :3]) if slot == 0 else original[2][:, :, :3])
                if not all(torch.equal(a, b) for a, b in zip((p, v, c), expected)):
                    raise ValueError('No-op teacher adapter is not exact: ' + family)
            proofs.append({'family': family, 'arm': arm, 'support_draw_sha256': info['support_draw_sha256'],
                           'label_sha256': info['tracks_sha256'], 'setting': info['teacher_setting']})
        if L.tensor_hash(model.state_dict()) != before:
            raise ValueError('Teacher weights mutated')
        del model
        torch.cuda.empty_cache()
    L.save_json(root / 'teacher-interface-smoke.json', {'complete': True, 'manifest_sha256': P.sha(root / 'manifest.json'),
                 'profiles': proofs, 'no_op_baselines_exact': True, 'query_times': [0, 29, 59]})
    print('GPU TEACHER INTERFACES PASSED all 24 family settings', flush=True)


def gpu_smoke(root):
    training = checked_training(root, 'smoke')
    import model as entry
    torch.manual_seed(P.BASE_SEED)
    video = torch.rand(1, 60, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [29., 32., 32.], [59., 48., 48.]]], device='cuda')
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(root / 'runtime.json')
    plan = get_plan(root, 'smoke')
    for arm in plan['pools']:
        m = entry.resources.setup_model(str(root / 'smoke/train/seed_0' / arm / 'cotracker_three_final.pth'), device='cuda')
        d = {}
        m.updateformer.feature_sidecar.diagnostics = d
        with torch.no_grad():
            output = m(video=video, queries=queries, iters=2)
        if m.feature_sidecar_kind != 'mamba' or m.updateformer.feature_sidecar.runtime != P.RUNTIME or not m.frozen_pretrained or d.get('sidecar_calls') != 2 or not d.get('sidecar_residual_rms_max') or any(not bool(torch.isfinite(t).all()) for t in output[:3]):
            raise ValueError('Trained student inference smoke failed')
        del m
        torch.cuda.empty_cache()
    L.save_json(root / 'smoke-gate.json', {'complete': True, 'manifest_sha256': P.sha(root / 'manifest.json'),
                'audit_sha256': P.sha(root / 'audits/smoke.json'), 'interface_sha256': P.sha(root / 'teacher-interface-smoke.json')})
    print('GPU TRAINED STUDENT SMOKE PASSED', flush=True)


def smoke_gate(root):
    checked_training(root, 'smoke')
    gate = P.read(root / 'smoke-gate.json')
    interface = P.read(root / 'teacher-interface-smoke.json')
    if (not gate['complete'] or gate['manifest_sha256'] != P.sha(root / 'manifest.json') or
        gate['audit_sha256'] != P.sha(root / 'audits/smoke.json') or gate['interface_sha256'] != P.sha(root / 'teacher-interface-smoke.json') or
        not interface['complete'] or not interface['no_op_baselines_exact'] or len(interface['profiles']) != 24 or
        interface['manifest_sha256'] != P.sha(root / 'manifest.json')):
        raise ValueError('Smoke gate incomplete')


def evaluation_inputs(root, plan, seed, arm):
    if arm == 'pretrained':
        checkpoint = root / 'pretrained-frozen.pth'
    else:
        if arm not in plan['pools'] or seed not in plan['seeds']:
            raise ValueError('Evaluation arm not in frozen plan')
        checkpoint = root / plan['phase'] / 'train' / f'seed_{seed}' / arm / 'cotracker_three_final.pth'
    return {'manifest_sha256': P.sha(root / 'manifest.json'), 'checkpoint_sha256': P.sha(checkpoint),
            'reference_images': P.read(root / 'reference-images.json'), 'runtime': P.RUNTIME,
            'student_config': P.read(root / 'manifest.json')['student_config'], 'seed': seed, 'arm': arm}


def bind_evaluation(root, stage, split, seed, arm):
    if (stage == 'test') != (split == 'test-38'):
        raise ValueError('Wrong evaluation split for stage')
    plan = get_plan(root, stage)
    relative = 'pretrained' if arm == 'pretrained' else f'seed_{seed}/{arm}'
    frozen(root / plan['phase'] / split / relative / 'evaluation-inputs.json', evaluation_inputs(root, plan, seed, arm))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'runtime', 'verify-arm', 'audit', 'report', 'select', 'teacher-smoke', 'gpu-smoke', 'smoke-gate', 'verify-training', 'plan-check', 'eval-bind'))
    p.add_argument('--root', type=Path, default=Path('/study'))
    p.add_argument('--prior', type=Path, default=Path('/prior'))
    p.add_argument('--package', type=Path, default=Path('/package'))
    p.add_argument('--weights', type=Path, default=Path('/checkpoints'))
    p.add_argument('--datasets', type=Path, default=Path('/datasets'))
    p.add_argument('--stage', choices=('smoke', 'pilot', 'confirm', 'combine', 'test'), default='smoke')
    p.add_argument('--split', choices=('validation-10', 'test-38'), default='validation-10')
    p.add_argument('--kind', choices=('Training', 'Inference'), default='Training')
    p.add_argument('--seed', type=int, choices=(-1, *P.SEEDS), default=0)
    p.add_argument('--arm', default='control')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.command == 'prepare': prepare(a.root, a.prior, a.package, a.weights, a.datasets)
    elif a.command == 'runtime': L.runtime(a.root, a.kind)
    elif a.command == 'verify-arm': verify_arm(a.root, get_plan(a.root, a.stage), a.seed, a.arm)
    elif a.command == 'audit': audit(a.root, a.stage)
    elif a.command == 'report': report(a.root, a.stage, a.split)
    elif a.command == 'select': select(a.root, a.stage)
    elif a.command == 'teacher-smoke': teacher_smoke(a.root, a.weights)
    elif a.command == 'gpu-smoke': gpu_smoke(a.root)
    elif a.command == 'smoke-gate': smoke_gate(a.root)
    elif a.command == 'verify-training': checked_training(a.root, a.stage)
    elif a.command == 'eval-bind': bind_evaluation(a.root, a.stage, a.split, a.seed, a.arm)
    else: get_plan(a.root, a.stage, deep=True)
