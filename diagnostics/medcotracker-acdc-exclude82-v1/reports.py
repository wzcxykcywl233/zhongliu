"""Paired three-seed reporting; test access requires a frozen validation decision."""
import argparse
import statistics
import math
from common import *
from summarize_backbone_growth import FIELDS, write_csv
from summarize_feature_sidecar import load_arm


def plans(root):
    result = {s: checked_plan(root / f'seed_{s}', expected_seed=s) for s in SEEDS}
    invariant = ('source_sha256', 'weights_sha256', 'images', 'teacher_pool',
                 'external_manifest_sha256', 'trackrad_ids', 'trackrad_files',
                 'student_config', 'prior_inventory_sha256', 'pilot_summary_sha256',
                 'pilot_selection_sha256')
    if any(result[s][k] != result[0][k] for s in SEEDS for k in invariant):
        raise ValueError('Seed plans use different study inputs')
    value = {'schema': 1, 'seeds': list(SEEDS), 'arms': list(ARMS),
             'external_steps': RULE['external_steps'], 'transfer_steps': RULE['transfer_steps'],
             'seed_manifests_sha256': {str(s): digest(root / f'seed_{s}/manifest.json') for s in SEEDS},
             'test_policy': 'Explicit test stage only; evaluate fixed R1/B1 and reference; never reselect on test'}
    S.frozen(root / 'manifest.json', value)
    return result


def training_proofs(root):
    values = plans(root)
    files = []
    for seed in SEEDS:
        local = root / f'seed_{seed}'
        proof = marker(local, 'full/training-audit.json')
        if (proof is None or proof['arms'] != list(ARMS) or not proof['paired_inputs_and_labels'] or
                not proof['frozen_backbones_exact'] or not proof['fresh_B_and_R1_branches_exact']):
            raise ValueError('All three paired full training audits required')
        files += [local / 'manifest.json', local / 'full/training-audit.json']
        files.extend(S.safe_file(local, r) for r in proof['files_sha256'])
    return values, files


def test_gate(root):
    training_proofs(root)
    proof = marker(root, 'reports/validation-10/verified.json')
    if proof is None or proof.get('split') != 'validation-10':
        raise ValueError('Complete frozen validation report required before test access')
    decision = read(root / 'reports/validation-10/selection.json')
    if decision['test_arms'] != ['pretrained', 'R1', 'B1'] or decision['selection_split'] != 'validation-10':
        raise ValueError('Test arms/selection changed')
    print('TEST GATE VERIFIED: frozen validation decision; no test selection', flush=True)
    return decision


def eval_bind(root, split, seed, arm):
    values, _ = training_proofs(root)
    if split == 'test-38':
        test_gate(root)
    if seed not in SEEDS or arm not in ('pretrained',) + ARMS or (arm == 'pretrained' and seed != 0):
        raise ValueError('Invalid evaluation arm/seed')
    local = root / f'seed_{seed}'
    checkpoint = local / ('pretrained-frozen.pth' if arm == 'pretrained' else f'full/train/{arm}/cotracker_three_final.pth')
    plan = values[seed]
    S.frozen(local / 'full' / split / arm / 'evaluation-inputs.json',
             {'manifest_sha256': digest(local / 'manifest.json'), 'checkpoint_sha256': digest(checkpoint),
              'profile': PROFILE, 'runtime': RUNTIME, 'split': split, 'images': plan['images'],
              'case_ids': plan['trackrad_ids'][split], 'seed': seed, 'arm': arm})


def test_data(root, prior, data):
    # Called only in an explicit -Stage test worker, after the CPU gate succeeds.
    test_gate(root)
    plan = checked_plan(root / 'seed_0', expected_seed=0)
    if digest(prior / 'frozen-study.json') != plan['prior_inventory_sha256']:
        raise ValueError('Historical dataset inventory changed')
    files = input_files(read(prior / 'frozen-study.json'), 'test-38')
    verify_dataset(data, files, plan['trackrad_ids']['test-38'])
    S.frozen(root / 'test-inputs.json', {'files_sha256': files,
                                      'selection_sha256': digest(root / 'reports/validation-10/selection.json')})
    print('TEST INPUTS VERIFIED: 38 cases, read-only', flush=True)


def choose(deltas):
    rows = [r for r in deltas if r['Candidate'] == 'B1' and r['Reference'] == 'R1']
    if len(rows) != 3 or {r['Seed'] for r in rows} != set(SEEDS):
        raise ValueError('Need exactly three paired seeds for selection')
    mean = {k: statistics.mean(r['Delta_' + k] for r in rows) for k in FIELDS}
    improved = sum(r['Delta_DSC'] > 0 for r in rows)
    worst = min(r['Delta_DSC'] for r in rows)
    passed = (mean['DSC'] > 0 and mean['D98'] >= -.002 and mean['HD95'] <= .02 and
              improved >= 2 and worst >= -.002)
    return {'selected': 'B1' if passed else 'R1', 'B1_qualified': passed,
            'paired_mean_deltas': mean, 'DSC_improved_seeds': improved,
            'worst_seed_DSC_delta': worst, 'selection_split': 'validation-10',
            'test_arms': ['pretrained', 'R1', 'B1'], 'test_requires_explicit_stage': True,
            'rule': 'Paired mean DSC>0, D98>=-.002, HD95<=.02; DSC improves in >=2/3 seeds; worst DSC>=-.002',
            'note': 'R1 is fallback. Both fixed arms may be reported on test, even if B1 fails; test cannot replace this decision.'}


def tables(records):
    means = []
    for arm in ('pretrained',) + ARMS:
        group = [r for r in records if r['Arm'] == arm]
        expected = 1 if arm == 'pretrained' else 3
        if len(group) != expected:
            raise ValueError('Wrong result seed coverage')
        row = {'Arm': arm, 'N': expected, 'ExternalSteps': 3000 if arm == 'B1' else 0,
               'TrackRADSteps': 0 if arm == 'pretrained' else 1000,
               'Parameters': 25385700 if arm == 'pretrained' else 28763025}
        for metric in (*FIELDS, 'TimeSec'):
            samples = [r[metric] for r in group]
            row[metric + '_Mean'] = statistics.mean(samples)
            row[metric + '_SD'] = statistics.stdev(samples) if expected > 1 else 0.
        means.append(row)
    pretrained = next(r for r in records if r['Arm'] == 'pretrained')
    paired = []
    for seed in SEEDS:
        arms = {r['Arm']: r for r in records if r['Seed'] == seed}
        arms['pretrained'] = pretrained
        for candidate, reference in (('B1', 'R1'), ('B1', 'pretrained'), ('R1', 'pretrained')):
            paired.append({'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                           **{'Delta_' + k: arms[candidate][k] - arms[reference][k] for k in FIELDS}})
    return means, paired


def compare_original(root, split, records, cases, values, baseline_report=Path('/baseline-report'),
                     baseline_metrics=Path('/baseline-metrics'), baseline_manifest=Path('/baseline-root-manifest.json')):
    """Compare STUDENTS after transfer, not teachers or external-only checkpoints."""
    receipt = read(baseline_report / 'verified.json')
    original = read(baseline_manifest)
    if (receipt.get('complete') is not True or receipt.get('split') != split or
            receipt.get('manifest_sha256') != digest(baseline_manifest) or
            original['seed_manifests_sha256'] != {str(s): values[s]['baseline_binding']['manifest.json'] for s in SEEDS}):
        raise ValueError('Historical report is not from the frozen baseline study')
    dest = root / 'reports' / split
    snapshot = dest / 'original-evidence'
    files, old_records, old_cases = [], [], {}
    for seed in SEEDS:
        for arm in ARMS:
            relative = f'seed_{seed}/full/{split}/{arm}/{PROFILE}/metrics.json'
            source = baseline_metrics / relative
            if digest(source) != receipt['files_sha256'].get(relative):
                raise ValueError('Historical official metrics changed: ' + relative)
            value = read(source)
            rows = value['results']
            by_case = {r['case_id']: r for r in rows}
            if len(by_case) != len(rows) or set(by_case) != set(values[seed]['trackrad_ids'][split]):
                raise ValueError('Historical case inventory mismatch')
            row = {'Split': split, 'Seed': seed, 'Arm': arm,
                   **{k: float(value['aggregates'][field]) for k, field in FIELDS.items()}}
            if not all(math.isfinite(row[k]) for k in FIELDS):
                raise ValueError('Nonfinite historical official metrics')
            old_records.append(row)
            old_cases[seed, arm] = by_case
            copied = snapshot / f'seed_{seed}-{arm}-metrics.json'
            S.frozen(copied, value)
            files.append(copied)
    paired, paired_cases = [], []
    for seed in SEEDS:
        for arm in ARMS:
            current = next(r for r in records if r['Seed'] == seed and r['Arm'] == arm)
            prior = next(r for r in old_records if r['Seed'] == seed and r['Arm'] == arm)
            candidate = 'B1_exclude082' if arm == 'B1' else 'R1_repeat'
            reference = 'B1_original80' if arm == 'B1' else 'R1_original'
            paired.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                           **{'Delta_' + k: current[k] - prior[k] for k in FIELDS}})
            for case in sorted(cases[seed, arm]):
                paired_cases.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference, 'Case': case,
                                     **{'Delta_' + k: float(cases[seed, arm][case][field]) - float(old_cases[seed, arm][case][field])
                                        for k, field in FIELDS.items()}})
    summary = []
    for candidate in ('B1_exclude082', 'R1_repeat'):
        group = [r for r in paired if r['Candidate'] == candidate]
        for metric in FIELDS:
            deltas = [r['Delta_' + metric] for r in group]
            summary.append({'Split': split, 'Candidate': candidate, 'Reference': group[0]['Reference'], 'Metric': metric,
                            'Seeds': 3, 'MeanDelta': statistics.mean(deltas), 'SD': statistics.stdev(deltas),
                            'ImprovedSeeds': sum(d > 0 if metric in ('DSC', 'D98') else d < 0 for d in deltas)})
    for name, rows in (('exclusion-vs-original-paired.csv', paired), ('exclusion-vs-original-summary.csv', summary),
                       ('exclusion-vs-original-cases.csv', paired_cases)):
        write_csv(dest / name, rows)
        files.append(dest / name)
    evidence = {'split': split, 'baseline_manifest_sha256': digest(baseline_manifest),
                'baseline_report_sha256': digest(baseline_report / 'verified.json'),
                'baseline_metrics_sha256': {r: h for r, h in receipt['files_sha256'].items() if r.endswith('/metrics.json')},
                'interpretation': 'Student performance after 3000 external + 1000 TrackRAD updates. Whole-patient exclusion with equal-budget RV replacement.',
                'limitations': 'Known duplicate074/076 deliberately unchanged. No external-dev selection. Public test repeatedly inspected; no blind test or p-values.'}
    S.frozen(snapshot / 'provenance.json', evidence)
    files.append(snapshot / 'provenance.json')
    for candidate in ('B1_exclude082', 'R1_repeat'):
        group = [r for r in paired if r['Candidate'] == candidate]
        print(f'RESULT {candidate} vs original: delta_DSC={statistics.mean(r["Delta_DSC"] for r in group):+.8f}; '
              f'delta_D98={statistics.mean(r["Delta_D98"] for r in group):+.8f}', flush=True)
    return files, paired


def report(root, split):
    values, files = training_proofs(root)
    if split == 'test-38':
        test_gate(root)
        if not (root / 'test-inputs.json').exists():
            raise ValueError('Test input hash audit missing')
    count = 10 if split == 'validation-10' else 38
    records, cases = [], {}
    jobs = [(0, 'pretrained')] + [(s, a) for s in SEEDS for a in ARMS]
    for seed, arm in jobs:
        eval_bind(root, split, seed, arm)
        local = root / f'seed_{seed}'
        folder = local / 'full' / split / arm
        plan = values[seed]
        if read(folder / 'frozen-images.json') != read(local / 'reference-images.json'):
            raise ValueError('Changed evaluation runtime')
        metrics, by_case, _, rms = load_arm(folder, 'none' if arm == 'pretrained' else 'mamba', count,
                                            25385700 if arm == 'pretrained' else 28763025)
        if set(by_case) != set(plan['trackrad_ids'][split]):
            raise ValueError('Wrong evaluation cases')
        for case in by_case:
            job = folder / PROFILE / 'checkpoint/jobs' / case
            diagnostic = read(job / 'diagnostics.json')
            if diagnostic['config'] != plan['student_config']:
                raise ValueError('Changed inference configuration')
            if arm != 'pretrained' and diagnostic['model']['sidecar_runtime'] != RUNTIME:
                raise ValueError('Changed sidecar runtime')
            files += [job / name for name in ('diagnostics.json', '.complete', 'output/images/mri-linac-series-targets/output.mha')]
        records.append({'Split': split, 'Seed': seed if arm != 'pretrained' else -1, 'Arm': arm,
                        'ResidualRmsMax': rms, **metrics})
        cases[(seed, arm)] = by_case
        files += [folder / PROFILE / 'metrics.json', folder / 'evaluation-inputs.json', folder / 'frozen-images.json']
    means, deltas = tables(records)
    means = [{'Split': split, **r} for r in means]
    deltas = [{'Split': split, **r} for r in deltas]
    paired_cases = [{'Split': split, 'Seed': s, 'Candidate': 'B1', 'Reference': 'R1', 'Case': case,
                     **{'Delta_' + k: float(cases[s, 'B1'][case][f]) - float(cases[s, 'R1'][case][f]) for k, f in FIELDS.items()}}
                    for s in SEEDS for case in sorted(cases[s, 'R1'])]
    dest = root / 'reports' / split
    dest.mkdir(parents=True, exist_ok=True)
    for name, rows in (('results.csv', records), ('summary.csv', means), ('paired-deltas.csv', deltas),
                       ('paired-cases.csv', paired_cases)):
        write_csv(dest / name, rows)
        files.append(dest / name)
    original_files, original_pairs = compare_original(root, split, records, cases, values)
    files += original_files
    if split == 'validation-10':
        decision = choose(deltas)
        comparison = [dict(r, Candidate='B1', Reference='R1') for r in original_pairs if r['Candidate'] == 'B1_exclude082']
        evidence = choose(comparison)
        decision['exclusion_vs_original_B1'] = {
            'candidate': 'B1_exclude082', 'reference': 'B1_original80',
            'passes_validation_guard': evidence['B1_qualified'],
            'paired_mean_deltas': evidence['paired_mean_deltas'],
            'DSC_improved_seeds': evidence['DSC_improved_seeds'],
            'worst_seed_DSC_delta': evidence['worst_seed_DSC_delta'], 'rule': evidence['rule']}
        drift = [r for r in original_pairs if r['Candidate'] == 'R1_repeat']
        decision['R1_repeat_max_abs_metric_drift'] = max(abs(r['Delta_' + k]) for r in drift for k in FIELDS)
        decision['R1_repeat_within_1e_5'] = decision['R1_repeat_max_abs_metric_drift'] <= 1e-5
        decision['interpretation'] = 'B1 in this report excludes patient082. Consult exclusion-vs-original files for the actual patient-removal effect.'
        decision['metrics_sha256'] = {p.relative_to(root).as_posix(): digest(p) for p in files if p.name == 'metrics.json'}
        decision['checkpoint_sha256'] = {p.relative_to(root).as_posix(): digest(p) for p in files if p.name == 'cotracker_three_final.pth'}
        S.frozen(dest / 'selection.json', decision)
        files.append(dest / 'selection.json')
    else:
        decision = read(root / 'reports/validation-10/selection.json')
        files += [root / 'test-inputs.json', root / 'reports/validation-10/selection.json']
    publish(root, f'reports/{split}/verified.json', files, split=split, seeds=list(SEEDS),
            selection_split='validation-10', selected=decision['selected'],
            note='Student metrics copied from official aggregates; SD is across 3 seeds, not independent patient uncertainty. Extra ACDC data/updates are not equal-compute. Test-38 is repeatedly inspected, not blind.')
    for row in means:
        print(f'RESULT {row["Arm"]}: DSC={row["DSC_Mean"]:.6f} +/- {row["DSC_SD"]:.6f}; D98={row["D98_Mean"]:.6f}', flush=True)
    print(f'FULL REPORT COMPLETE {split}; validation decision={decision["selected"]}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('freeze-plan', 'test-gate', 'test-data', 'eval-bind', 'report'))
    p.add_argument('--split', choices=('validation-10', 'test-38'), default='validation-10')
    p.add_argument('--seed', type=int, choices=SEEDS, default=0)
    p.add_argument('--arm', choices=('pretrained',) + ARMS, default='pretrained')
    args = p.parse_args()
    root = Path('/study-all')
    if args.command == 'freeze-plan':
        plans(root)
        print('PLAN FROZEN: 3 paired seeds, B1/R1, 3000/1000 steps', flush=True)
    elif args.command == 'test-gate': test_gate(root)
    elif args.command == 'test-data': test_data(root, Path('/prior'), Path('/test-data'))
    elif args.command == 'eval-bind': eval_bind(root, args.split, args.seed, args.arm)
    else: report(root, args.split)
