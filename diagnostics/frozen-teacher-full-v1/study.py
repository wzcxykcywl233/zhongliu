"""Freeze the validated teacher policy and retrain the best existing student.

Runs inside the *existing* ts-v1 images. No change to the old study or trainer.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path, PureWindowsPath
import statistics
import sys

APP = Path(os.environ.get('COTRACKER_APP_DIR', '/opt/app'))
sys.path[:0] = [str(APP / 'ext/co-tracker'), str(APP), str(APP / 'experiments')]
import torch
from audit_backbone_growth import file_sha, validate_checkpoint
from cotracker.models.core.cotracker.backbone_growth import save_json
from cotracker.utils.teacher_sampling import TeacherPairSampler
from cotracker.utils.offline_teacher_policy import SETTINGS
from experiments import get_experiment_config
from summarize_backbone_growth import FIELDS, load_arm as load_pilot_arm, write_csv
from teacher_student_study import BASE_SEED, INPUT_KEYS, LABEL_KEYS, read_records, verify_training

POLICIES = ('old', 'iter6')
SEEDS = (0, 1, 2)
STEPS = 1000
PROFILE = 'rt_points_0'
OLD_PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def within(root, relative):
    relative = str(relative).replace('\\', '/')
    if relative.startswith('/') or ':' in relative or '..' in relative.split('/'):
        raise ValueError('unsafe artifact path')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('artifact escaped input root')
    return path


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen selection/protocol changed; preserve results and use a NEW OutputRoot: ' + str(path))
    else:
        save_json(path, value)


def checked_hashes(root, hashes):
    for relative, expected in hashes.items():
        if file_sha(within(root, relative)) != expected.lower():
            raise ValueError('recorded artifact changed: ' + str(relative))


def student_config():
    cfg = get_experiment_config(PROFILE)[1]
    parent = get_experiment_config(OLD_PROFILE)[1]
    if cfg != replace(parent, border_points=250) or cfg != get_experiment_config('pd_sparse_native')[1]:
        raise ValueError('best-student 250-point profile no longer matches historical configuration')
    return asdict(cfg)


def verify_datasets(protocol, datasets):
    expected_counts = {'train-40': ('train', 40), 'validation-10': ('validation', 10), 'test-38': ('test', 38)}
    seen, result = set(), {}
    for split, (alias, count) in expected_counts.items():
        rows = [r for r in protocol['Dataset'] if r['Split'] == split]
        cases = sorted({r['Case'] for r in rows})
        root = Path(datasets) / alias
        if len(cases) != count or seen.intersection(cases) or sorted(p.name for p in root.iterdir() if p.is_dir()) != cases:
            raise ValueError('dataset counts/IDs changed or splits overlap: ' + split)
        seen.update(cases)
        inventory = set()
        for row in rows:
            parts = PureWindowsPath(row['File']).parts
            offsets = [i for i, part in enumerate(parts) if part == row['Case']]
            if len(offsets) != 1:
                raise ValueError('ambiguous original dataset path')
            relative = '/'.join(parts[offsets[0]:])
            if relative in inventory:
                raise ValueError('duplicate dataset inventory entry')
            inventory.add(relative)
            if file_sha(within(root, relative)) != row['SHA256'].lower():
                raise ValueError('dataset file changed: ' + relative)
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if inventory != actual:
            raise ValueError('dataset file inventory changed: ' + split)
        result[split] = cases
    return result


def choose_teacher(scores):
    candidates = ('scaled', 'grid0', 'iter6')
    if set(scores) != {'pretrained', 'old', *candidates}:
        raise ValueError('expected the complete four-arm pilot plus pretrained reference')
    for values in scores.values():
        if any(not math.isfinite(float(values[k])) for k in FIELDS):
            raise ValueError('nonfinite pilot metric')
    return min(candidates, key=lambda p: (-scores[p]['DSC'], scores[p]['HD95'], scores[p]['CD'], -scores[p]['D98'], p))


def prepare(prior, root, checkout, package, weights, datasets):
    """Read-only verification of ts-v1; only the new output directory is written."""
    protocol = read(prior / 'frozen-study.json')
    if (protocol['FullSteps'] != STEPS or protocol['FullSeeds'] != list(SEEDS) or
        protocol['InferenceProfile'] != OLD_PROFILE or protocol['SequenceLength'] != 10 or
        protocol['QueryPoints'] != 384 or protocol['TrainIterations'] != 4 or
        protocol['LearningRate'] != .00005 or protocol['BaseSeed'] != BASE_SEED or
        protocol['ConfidenceTarget'] != 'hard' or protocol['AuxiliaryWeight'] != 0 or protocol['MaskWeight'] != 0 or
        protocol['TeacherPool'] != ['cotracker2v1','online_cotracker_three','offline_cotracker_three']):
        raise ValueError('prior study is not the audited teacher-student recipe')
    checked_hashes(checkout, {r['File']: r['SHA256'] for r in protocol['Source']})
    checked_hashes(weights, {r['Name']: r['SHA256'] for r in protocol['Weights']})
    case_ids = verify_datasets(protocol, datasets)
    images = read(prior / 'local-runtime-images.json')
    for name in ('Training', 'Inference', 'Evaluation'):
        value = images[name]
        if not isinstance(value, str) or len(value) != 71 or not value.startswith('sha256:') or any(c not in '0123456789abcdef' for c in value[7:]):
            raise ValueError('installed immutable image ID required')
    reference = read(prior / 'reference-images.json')
    if reference != {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']}:
        raise ValueError('prior image identities disagree')
    for proof in ('gpu-teacher-proof.json', 'gpu-student-inference-proof.json', 'training-audit.json'):
        if not read(prior / 'smoke' / proof)['complete']:
            raise ValueError('original GPU smoke is incomplete')
    verify_training(prior / 'pilot', [0], 125)
    report = read(prior / 'pilot/validation-10/teacher-student-audit.json')
    if (not report['complete'] or report['split'] != 'validation-10' or report['cases'] != 10 or
        report['seeds'] != [0] or report['steps'] != 125 or report['inference_profile'] != OLD_PROFILE or
        report['policies'] != ['old','scaled','grid0','iter6']):
        raise ValueError('pilot validation provenance mismatch')
    checked_hashes(prior / 'pilot/validation-10', report['metrics_sha256'])
    scores = {}
    for policy in ('pretrained', 'old', 'scaled', 'grid0', 'iter6'):
        relative = 'pretrained' if policy == 'pretrained' else 'seed_0/' + policy
        folder = prior / 'pilot/validation-10' / relative
        aggregate, cases, _ = load_pilot_arm(folder, 'base', 10)
        if set(cases) != set(case_ids['validation-10']) or read(folder / 'frozen-images.json') != reference:
            raise ValueError('pilot evaluated cases/runtime changed')
        scores[policy] = aggregate
    selected = choose_teacher(scores)
    if selected != 'iter6':
        raise ValueError('supplied pilot no longer selects the requested iter6 teacher; do not silently switch experiments')
    cfg = student_config()
    package_hashes = {p.name: file_sha(p) for p in sorted(package.iterdir()) if p.suffix in ('.py', '.ps1', '.txt')}
    selection = {
        'schema': 1, 'selected_policy': selected, 'selection_split': 'validation-10',
        'selection_kind': 'student pilot performance, NOT teacher-generator metrics',
        'rule': 'Best new-policy pilot DSC; ties HD95,CD,D98,name. No test selection.',
        'pilot_scores': scores, 'pilot_metrics_sha256': report['metrics_sha256'],
        'pilot_report_sha256': file_sha(prior / 'pilot/validation-10/teacher-student-audit.json'),
        'prior_evidence_sha256': {name:file_sha(prior / name) for name in (
            'smoke/gpu-teacher-proof.json','smoke/gpu-student-inference-proof.json','smoke/training-audit.json',
            'pilot/training-audit.json','local-runtime-images.json','reference-images.json')},
        'teacher_checkpoint': SETTINGS[selected][0], 'teacher_support_points': SETTINGS[selected][1],
        'teacher_iterations': SETTINGS[selected][2], 'teacher_weights_sha256': {r['Name']: r['SHA256'].lower() for r in protocol['Weights']},
        'teacher_update': 'none: frozen checkpoint/eval/no_grad; only offline teacher slot is changed',
        'teacher_pool': protocol['TeacherPool'], 'student_initialization': 'scaled_offline.pth, NOT a pilot-trained checkpoint',
        'student_architecture': 'base', 'student_parameters': 25385700,
        'student_inference_profile': PROFILE, 'student_inference_config': cfg,
        'training': {'steps': STEPS, 'seeds': list(SEEDS), 'query_points': 384, 'frames': 10, 'student_iterations': 4,
                     'lr': .00005, 'base_seed': BASE_SEED, 'confidence_target': 'hard', 'auxiliary_weight': 0, 'mask_weight': 0},
        'comparison_policies': list(POLICIES), 'images': {k: images[k] for k in ('Training','Inference','Evaluation')},
        'case_ids': case_ids, 'prior_protocol_sha256': file_sha(prior / 'frozen-study.json'),
        'package_sha256': package_hashes, 'max_median_step_seconds': protocol['MaxMedianStepSeconds'],
        'arm_timeout_seconds': protocol['ArmTimeoutSeconds'],
        'user_requested_full_run_despite_pilot_degradation': True,
        'selected_pilot_below_pretrained': scores[selected]['DSC'] < scores['pretrained']['DSC'],
        'limitations': 'This is full 1000-step retraining, not fine-tuning the teacher or extending 125-step checkpoints. '
                       '250 is inference boundary count, not training query count. Test-38 is repeatedly inspected, not blind.'}
    frozen(root / 'selection.json', selection)
    frozen(root / 'reference-images.json', reference)
    for seed in SEEDS:
        for policy in POLICIES:
            folder = root / 'train' / f'seed_{seed}' / policy
            folder.mkdir(parents=True, exist_ok=True)
            frozen(folder / 'run-config.json', {'policy': policy, 'offline_checkpoint': SETTINGS[policy][0],
                   'steps': STEPS, 'seed': seed, 'training_seed': BASE_SEED + seed,
                   'selection_sha256': file_sha(root / 'selection.json'), 'sequence_len': 10, 'queries': 384,
                   'train_iters': 4, 'lr': .00005, 'save_every_steps': 25, 'phase': 'full'})
    print('FROZEN iter6 TEACHER; best existing base student=rt_points_0 (250 boundary points); old/iter6 x3x1000', flush=True)
    return selection


def runtime(root, prior, kind):
    selection = read(root / 'selection.json')
    protocol = read(prior / 'frozen-study.json')
    expected = {r['File']: r['SHA256'].lower() for r in protocol['Source']}
    # Verify the executable copied source, not merely an image tag or wheel name.
    names = ('ext/co-tracker/train_on_real_data.py', 'ext/co-tracker/cotracker/utils/offline_teacher_policy.py',
             'ext/co-tracker/cotracker/datasets/trackrad_video_dataset.py', 'ext/co-tracker/cotracker/utils/resumable_loader.py',
             'experiments.py', 'parameter_retune.py', 'model.py', 'resources/model.py')
    hashes = {}
    for relative in names:
        digest = file_sha(APP / relative)
        if digest != expected['cotracker-algorithm/' + relative]:
            raise ValueError('runtime differs from frozen pilot code: ' + relative)
        hashes[relative] = digest
    if student_config() != selection['student_inference_config']:
        raise ValueError('runtime student profile changed')
    frozen(root / ('runtime-' + kind + '.json'), {'complete': True, 'kind': kind, 'code_sha256': hashes})
    print('FROZEN RUNTIME VERIFIED ' + kind, flush=True)


def verify_arm(root, seed, policy):
    selection = read(root / 'selection.json')
    folder = root / 'train' / f'seed_{seed}' / policy
    cfg = read(folder / 'run-config.json')
    if cfg != {'policy': policy, 'offline_checkpoint': SETTINGS[policy][0], 'steps': STEPS, 'seed': seed,
               'training_seed': BASE_SEED + seed, 'selection_sha256': file_sha(root / 'selection.json'),
               'sequence_len': 10, 'queries': 384, 'train_iters': 4, 'lr': .00005, 'save_every_steps': 25, 'phase': 'full'}:
        raise ValueError('training config changed')
    init = read(folder / 'initialization-report.json')
    if init['architecture'] != 'base' or not init['initialization_outputs_exact'] or init['added_parameters']:
        raise ValueError('student initialization changed')
    checkpoint = torch.load(folder / 'cotracker_three_final.pth', map_location='cpu', weights_only=True)
    if checkpoint.get('teacher_student_policy') != policy or checkpoint.get('feature_sidecar') != 'none':
        raise ValueError('wrong trained student/teacher provenance')
    validate_checkpoint(folder, 'base', STEPS)
    rows = read_records(folder / 'paired-steps.jsonl', STEPS, policy)
    sampler = TeacherPairSampler(3, seed=BASE_SEED + seed)
    if any(rows[i]['primary_index'] != sampler.sample().primary for i in range(STEPS)):
        raise ValueError('teacher sampling sequence changed')
    median = statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2)
    if median > selection['max_median_step_seconds']:
        raise ValueError('step runtime exceeded the original safety budget')
    print(f'VERIFIED seed_{seed}/{policy}: 1000 steps; median={median:.3f}s', flush=True)
    return rows


def audit(root):
    torch.set_num_threads(1)
    hashes, artifacts, proofs, costs = {}, {}, [], []
    for seed in SEEDS:
        reference = None
        for policy in POLICIES:
            rows = verify_arm(root, seed, policy)
            if reference is None:
                reference = rows
            else:
                for i, row in rows.items():
                    ref = reference[i]
                    if any(row[k] != ref[k] for k in INPUT_KEYS) or any(row['label_audit'][k] != ref['label_audit'][k] for k in ('support_draw_sha256','query_time_counts')):
                        raise ValueError('paired training input changed')
                    if row['primary_index'] != 2 and any(row['label_audit'][k] != ref['label_audit'][k] for k in LABEL_KEYS):
                        raise ValueError('unmodified teacher labels changed')
            folder = root / 'train' / f'seed_{seed}' / policy
            final = folder / 'cotracker_three_final.pth'
            hashes[final.relative_to(root).as_posix()] = file_sha(final)
            for name in ('run-config.json','paired-steps.jsonl','initialization-report.json'):
                path = folder / name
                artifacts[path.relative_to(root).as_posix()] = file_sha(path)
            offline = [r for r in rows.values() if r['primary_index'] == 2]
            if not offline:
                raise ValueError('offline intervention was never sampled')
            proofs.append({'seed': seed, 'policy': policy, 'paired_steps': STEPS, 'offline_steps': len(offline),
                           'median_step_seconds': statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2)})
            for slot in (0,1,2):
                subset = [r for r in rows.values() if r['primary_index'] == slot]
                costs.append({'Seed': seed, 'Policy': policy, 'TeacherSlot': slot, 'Steps': len(subset),
                              'TeacherSecondsMean': statistics.mean(r['label_audit']['teacher_seconds'] for r in subset),
                              'StepSecondsMean': statistics.mean(r['step_seconds'] for r in subset)})
    result = {'complete': True, 'steps': STEPS, 'seeds': list(SEEDS), 'policies': list(POLICIES), 'proofs': proofs,
              'selection_sha256': file_sha(root / 'selection.json'), 'checkpoints_sha256': hashes, 'artifacts_sha256': artifacts}
    save_json(root / 'training-audit.json', result)
    write_csv(root / 'training-runtime.csv', costs)
    print('FULL PAIRED TRAINING AUDIT PASSED', flush=True)


def checked_training(root):
    report = read(root / 'training-audit.json')
    if (not report['complete'] or report['steps'] != STEPS or report['seeds'] != list(SEEDS) or
        report['policies'] != list(POLICIES) or report['selection_sha256'] != file_sha(root / 'selection.json')):
        raise ValueError('training audit does not match frozen experiment')
    expected = {f'train/seed_{seed}/{policy}/cotracker_three_final.pth' for seed in SEEDS for policy in POLICIES}
    if set(report['checkpoints_sha256']) != expected or len(report['proofs']) != 6:
        raise ValueError('full training audit does not cover all six predeclared arms')
    checked_hashes(root, report['checkpoints_sha256'])
    checked_hashes(root, report['artifacts_sha256'])
    return report


def load_student_arm(folder, expected_cases, config, expected_images=None):
    if expected_images is not None and read(folder / 'frozen-images.json') != expected_images:
        raise ValueError('evaluation runtime differs from the frozen study')
    metrics_path = folder / PROFILE / 'metrics.json'
    metrics = read(metrics_path)
    cases = {r['case_id']: r for r in metrics['results']}
    if len(metrics['results']) != len(cases) or set(cases) != set(expected_cases):
        raise ValueError('official student case IDs/coverage mismatch')
    outputs = {}
    for case, row in cases.items():
        if any(not math.isfinite(float(row[k])) for k in FIELDS.values()):
            raise ValueError('nonfinite official case metric')
        job = folder / PROFILE / 'checkpoint/jobs' / case
        d = read(job / 'diagnostics.json')
        if d['profile'] != PROFILE or d['config'] != config or d['model'] != {'backbone_architecture':'base','parameters':25385700}:
            raise ValueError('student architecture/profile was not the frozen best configuration')
        digest = file_sha(job / 'output/images/mri-linac-series-targets/output.mha')
        if not (job / '.complete').is_file() or digest != d['prediction']['output_file_sha256']:
            raise ValueError('committed student output changed')
        outputs[(job / 'output/images/mri-linac-series-targets/output.mha').relative_to(folder).as_posix()] = digest
    aggregate = {k: float(metrics['aggregates'][v]) for k,v in FIELDS.items()}
    aggregate['TimeSec'] = float(metrics['aggregates']['total_time'])
    if any(not math.isfinite(x) for x in aggregate.values()):
        raise ValueError('nonfinite official aggregate')
    return aggregate, cases, file_sha(metrics_path), outputs


def report(root, split):
    checked_training(root)
    selection = read(root / 'selection.json')
    expected = selection['case_ids'][split]
    folder = root / split
    images = read(root / 'reference-images.json')
    ref, ref_cases, digest, outputs = load_student_arm(folder / 'pretrained', expected, selection['student_inference_config'], images)
    records = [{'Split':split, 'Seed':-1, 'Policy':'pretrained', **ref}]
    hashes = {f'pretrained/{PROFILE}/metrics.json':digest}
    output_hashes = {'pretrained/' + k:v for k,v in outputs.items()}
    values, pairs, case_pairs = {}, [], []
    for seed in SEEDS:
        arms = {'pretrained':(ref,ref_cases)}
        for policy in POLICIES:
            relative = f'seed_{seed}/{policy}'
            aggregate, cases, digest, outputs = load_student_arm(folder / relative, expected, selection['student_inference_config'], images)
            arms[policy] = aggregate, cases
            values.setdefault(policy, []).append(aggregate)
            records.append({'Split':split, 'Seed':seed, 'Policy':policy, **aggregate})
            hashes[f'{relative}/{PROFILE}/metrics.json'] = digest
            output_hashes.update({relative + '/' + k:v for k,v in outputs.items()})
        for candidate, reference in (('iter6','old'),('old','pretrained'),('iter6','pretrained')):
            a, ac = arms[reference]
            b, bc = arms[candidate]
            pairs.append({'Split':split, 'Seed':seed, 'Candidate':candidate, 'Reference':reference,
                          **{f'Delta_{k}': b[k]-a[k] for k in FIELDS}})
            for case in sorted(expected):
                case_pairs.append({'Split':split, 'Seed':seed, 'Candidate':candidate, 'Reference':reference, 'Case':case,
                                   **{f'Delta_{k}':float(bc[case][v])-float(ac[case][v]) for k,v in FIELDS.items()}})
    means = []
    for policy, samples in [('pretrained',[ref]), *values.items()]:
        row = {'Split':split, 'Policy':policy, 'N':len(samples), 'Parameters':25385700, 'StudentProfile':PROFILE}
        for k in (*FIELDS,'TimeSec'):
            row[k+'_Mean'] = statistics.mean(r[k] for r in samples)
            row[k+'_SD'] = statistics.stdev(r[k] for r in samples) if len(samples)>1 else 0.
        means.append(row)
    pair_means = []
    for candidate, reference in (('iter6','old'),('old','pretrained'),('iter6','pretrained')):
        subset = [r for r in pairs if (r['Candidate'],r['Reference']) == (candidate,reference)]
        row = {'Split':split, 'Candidate':candidate, 'Reference':reference, 'N':len(subset)}
        for k in FIELDS:
            samples = [r['Delta_'+k] for r in subset]
            row['Delta_'+k+'_Mean'] = statistics.mean(samples)
            row['Delta_'+k+'_SD'] = statistics.stdev(samples)
        pair_means.append(row)
    for name, rows in (('results',records),('summary',means),('paired-deltas',pairs),('paired-summary',pair_means),('paired-cases',case_pairs)):
        write_csv(folder / ('frozen-teacher-student-'+name+'.csv'), rows)
    save_json(folder / 'audit.json', {'complete':True, 'split':split, 'cases':len(expected), 'steps':STEPS,
              'seeds':list(SEEDS), 'policies':list(POLICIES), 'student_profile':PROFILE,
              'selection_sha256':file_sha(root / 'selection.json'), 'metrics_sha256':hashes, 'outputs_sha256':output_hashes,
              'note':'Official student metrics. Pretrained N=1 is not three independent references. '
                     'Teacher frozen from validation only; no test-based selection; public test already inspected.'})
    print(json.dumps(means), flush=True)


def test_gate(root):
    checked_training(root)
    proof = read(root / 'validation-10/audit.json')
    if (not proof['complete'] or proof['split'] != 'validation-10' or proof['cases'] != 10 or
        proof['selection_sha256'] != file_sha(root / 'selection.json') or proof['student_profile'] != PROFILE or
        proof['steps'] != STEPS or proof['seeds'] != list(SEEDS) or proof['policies'] != list(POLICIES) or
        len(proof['metrics_sha256']) != 7):
        raise ValueError('full validation must complete before test')
    checked_hashes(root / 'validation-10', proof['metrics_sha256'])
    checked_hashes(root / 'validation-10', proof['outputs_sha256'])
    frozen(root / 'test-shortlist.json', {'policies':list(POLICIES), 'seeds':list(SEEDS), 'steps':STEPS, 'student_profile':PROFILE,
           'validation_audit_sha256':file_sha(root / 'validation-10/audit.json'), 'rule':'Report both predeclared policies, even if degraded. No test-driven selection.'})
    print('FULL VALIDATION VERIFIED; frozen 38-case report may proceed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare','runtime','verify-arm','audit','verify-training','report','test-gate'))
    parser.add_argument('--root', type=Path, default=Path('/results'))
    parser.add_argument('--prior', type=Path, default=Path('/prior'))
    parser.add_argument('--checkout', type=Path, default=Path('/checkout'))
    parser.add_argument('--package', type=Path, default=Path('/package'))
    parser.add_argument('--weights', type=Path, default=Path('/checkpoints'))
    parser.add_argument('--datasets', type=Path, default=Path('/datasets'))
    parser.add_argument('--kind', choices=('Training','Inference'), default='Training')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    parser.add_argument('--policy', choices=POLICIES, default='old')
    parser.add_argument('--split', choices=('validation-10','test-38'), default='validation-10')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.prior,args.root,args.checkout,args.package,args.weights,args.datasets)
    elif args.command == 'runtime':
        runtime(args.root,args.prior,args.kind)
    elif args.command == 'verify-arm':
        torch.set_num_threads(1)
        verify_arm(args.root,args.seed,args.policy)
    elif args.command == 'audit':
        audit(args.root)
    elif args.command == 'verify-training':
        checked_training(args.root)
    elif args.command == 'test-gate':
        test_gate(args.root)
    else:
        report(args.root,args.split)
