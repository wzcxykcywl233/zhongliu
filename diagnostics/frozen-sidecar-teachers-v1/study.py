"""Paired frozen-Mamba student audits and official 10/38-case reporting."""
import argparse
from dataclasses import asdict
import importlib.util
import math
import os
from pathlib import Path
import statistics
import sys

from protocol import (BASE_SEED, KIND, PAIRS, PARAMETERS, POLICIES, PROFILE, RUNTIME,
                      SEEDS, SETTINGS, SMOKE_STEPS, TRAINABLE, compare_rows, read, read_records, sha, train_config)

APP = Path(os.environ.get('COTRACKER_APP_DIR', '/opt/app'))
CHECKOUT = Path(os.environ.get('SIDECAR_CHECKOUT', '/checkout'))
sys.path[:0] = [str(APP / 'ext/co-tracker'), str(APP), str(APP / 'experiments')]
import torch
from cotracker.models.core.cotracker.backbone_growth import save_json, state_from_checkpoint
from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, tensor_hash
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.utils.teacher_sampling import TeacherPairSampler
from experiments import get_experiment_config
from freeze_sidecar_reference import freeze
from summarize_feature_sidecar import load_arm
from summarize_backbone_growth import FIELDS, write_csv

spec = importlib.util.spec_from_file_location('frozen_full_helpers', CHECKOUT / 'diagnostics/frozen-teacher-full-v1/study.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
checked_hashes = helpers.checked_hashes


def frozen(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    helpers.frozen(path, value)


def prepare(root, prior, package, weights, datasets):
    protocol = read(prior / 'frozen-study.json')
    checked_hashes(CHECKOUT, {r['File']: r['SHA256'] for r in protocol['Source']})
    checked_hashes(weights, {r['Name']: r['SHA256'] for r in protocol['Weights']})
    case_ids = helpers.verify_datasets(protocol, datasets)
    images = read(prior / 'local-runtime-images.json')
    for name in ('Training', 'Inference', 'Evaluation'):
        value = images[name]
        if len(value) != 71 or not value.startswith('sha256:') or any(c not in '0123456789abcdef' for c in value[7:]):
            raise ValueError('Immutable local image ID required')
    reference = {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']}
    if read(prior / 'reference-images.json') != reference:
        raise ValueError('Image identities disagree')
    config = asdict(get_experiment_config(PROFILE)[1])
    if config['border_points'] != 1000 or config['n_iterations'] != 2:
        raise ValueError('Historical frozen-student inference profile changed')
    manifest = {'schema': 1, 'direction': 'Freeze official CoTracker backbone; train the fixed Mamba branch; compare teachers only.',
                'policies': list(POLICIES), 'teacher_settings': {k: list(v) for k, v in SETTINGS.items()},
                'kind': KIND, 'seeds': list(SEEDS), 'steps': 1000, 'smoke_steps': SMOKE_STEPS, 'base_seed': BASE_SEED,
                'sequence_len': 60, 'traj_per_sample': 128, 'train_iters': 4, 'lr': .00005,
                'parameters': PARAMETERS, 'trainable_parameters': TRAINABLE, 'runtime': RUNTIME,
                'student_profile': PROFILE, 'student_config': config, 'case_ids': case_ids,
                'source_sha256': {r['File']: r['SHA256'].lower() for r in protocol['Source']},
                'weights_sha256': {r['Name']: r['SHA256'].lower() for r in protocol['Weights']},
                'prior_protocol_sha256': sha(prior / 'frozen-study.json'),
                'images': {k: images[k] for k in ('Training', 'Inference', 'Evaluation')},
                'trainer_sha256': sha(CHECKOUT / 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'),
                'package_sha256': {p.name: sha(p) for p in package.iterdir() if p.is_file() and p.suffix in ('.py', '.ps1', '.txt')},
                'helper_sha256': sha(CHECKOUT / 'diagnostics/frozen-teacher-full-v1/study.py'),
                'comparisons': [list(p) for p in PAIRS],
                'rule': 'All four teacher policies are fixed before this experiment. Report all, including degraded arms; no test selection.',
                'limitations': 'Compact scan changes execution, not the branch design. Fresh old-teacher controls are mandatory. '
                               'Only offline teacher slot changes; other two teachers stay fixed. Public test-38 is already inspected.'}
    frozen(root / 'manifest.json', manifest)
    frozen(root / 'runtime.json', RUNTIME)
    frozen(root / 'reference-images.json', reference)
    freeze(weights / 'scaled_offline.pth', root / 'pretrained-frozen.pth')
    state = state_from_checkpoint(torch.load(weights / 'scaled_offline.pth', map_location='cpu', weights_only=True))
    frozen(root / 'reference-proof.json', {'base_tensor_sha256': tensor_hash(state),
           'checkpoint_sha256': sha(root / 'pretrained-frozen.pth')})
    for phase, seeds in (('smoke', (0,)), ('full', SEEDS)):
        for seed in seeds:
            for policy in POLICIES:
                frozen(root / phase / 'train' / f'seed_{seed}' / policy / 'run-config.json',
                       train_config(seed, policy, phase == 'smoke', sha(root / 'manifest.json')))
    print('PLAN VERIFIED: frozen Mamba; old/scaled/grid0/iter6; 60 frames / 128 queries; 40/10/38 disjoint', flush=True)


def verify_runtime_sources(app, source_hashes):
    # Mirror Dockerfile.training.reuse and Dockerfile.inference.reuse COPY scope.
    # prepare() still checks the entire checkout inventory, including top-level tests.
    # Never skip a missing file within a copied tree: that indicates a stale image.
    hashes = {}
    prefix = 'cotracker-algorithm/'
    copied_trees = ('ext/co-tracker/', 'resources/', 'experiments/')
    for relative, digest in source_hashes.items():
        if not relative.startswith(prefix) or not relative.endswith('.py'):
            continue
        inside = relative[len(prefix):]
        if '/' in inside and not inside.startswith(copied_trees):
            continue
        path = app / inside
        if not path.is_file():
            raise ValueError('Runtime source missing from pinned image: ' + relative)
        if sha(path) != digest:
            raise ValueError('Runtime source mismatch: ' + relative)
        hashes[relative] = digest
    if not hashes:
        raise ValueError('Runtime source inventory is empty')
    return hashes


def runtime(root, kind):
    manifest = read(root / 'manifest.json')
    hashes = verify_runtime_sources(APP, manifest['source_sha256'])
    from cotracker.models.core.cotracker import feature_sidecar
    if APP.resolve() not in Path(feature_sidecar.__file__).resolve().parents:
        raise ValueError('Wrong installed CoTracker module loaded')
    frozen(root / ('runtime-' + kind + '.json'), {'complete': True,
           'scope': 'reuse-dockerfiles-python-copy-scope-v2', 'source_sha256': hashes})
    print('RUNTIME VERIFIED ' + kind, flush=True)


def phase_seeds(phase):
    return (0,) if phase == 'smoke' else SEEDS


def verify_arm(root, phase, seed, policy):
    folder = root / phase / 'train' / f'seed_{seed}' / policy
    expected = train_config(seed, policy, phase == 'smoke', sha(root / 'manifest.json'))
    if read(folder / 'run-config.json') != expected:
        raise ValueError('Arm configuration changed')
    init = read(folder / 'initialization-report.json')
    info = init['sidecar']
    base_hash = read(root / 'reference-proof.json')['base_tensor_sha256']
    if (init['architecture'] != 'base' or not init['initialization_outputs_exact'] or
        info['kind'] != KIND or not info['zero_projection_verified'] or
        not info['initialization_outputs_exact'] or any(info['initialization_max_abs_difference']) or
        info['initialization_reference'] != 'frozen-pretrained' or info['base_tensor_sha256'] != base_hash or
        info['execution_runtime'] != RUNTIME or info['parameters'] != PARAMETERS or info['trainable_parameters'] != TRAINABLE):
        raise ValueError('Frozen initialization/parameter proof failed')
    adapter = read(folder / 'trainer-adapter.json')
    manifest = read(root / 'manifest.json')
    if (adapter['source_sha256'] != manifest['trainer_sha256'] or
        adapter['adapter_sha256'] != manifest['package_sha256']['train.py'] or
        adapter['protocol_sha256'] != manifest['package_sha256']['protocol.py'] or
        adapter['stop_steps'] != expected['steps'] or adapter['scheduler_steps'] != 1000):
        raise ValueError('Executed adapter changed')
    final = torch.load(folder / 'cotracker_three_final.pth', map_location='cpu', weights_only=True)
    if (final['total_steps'] != expected['steps'] or final['feature_sidecar'] != KIND or
        final['backbone_architecture'] != 'base' or final['teacher_student_policy'] != policy):
        raise ValueError('Wrong trained checkpoint metadata')
    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    attach_sidecar(model, KIND)
    model.load_state_dict(final['model'], strict=True)
    if (sum(p.numel() for p in model.parameters()) != PARAMETERS or
        sum(p.numel() for p in model.parameters() if p.requires_grad) != TRAINABLE):
        raise ValueError('Actual parameter budget changed')
    if tensor_hash(model.state_dict()) != base_hash:
        raise ValueError('Frozen backbone tensors changed')
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite model parameters')
    if not bool(model.updateformer.feature_sidecar.output.weight.count_nonzero()):
        raise ValueError('Side branch did not learn')
    rows = read_records(folder / 'paired-steps.jsonl', expected['steps'], policy)
    sampler = TeacherPairSampler(3, seed=BASE_SEED + seed)
    if any(rows[i]['primary_index'] != sampler.sample().primary for i in range(expected['steps'])):
        raise ValueError('Teacher sampling schedule changed')
    print(f'VERIFIED {phase}/seed_{seed}/{policy}: backbone exact, steps={len(rows)}', flush=True)
    return rows, info['initial_sidecar_sha256']


def audit(root, phase):
    proofs, artifacts, costs = [], {}, []
    for seed in phase_seeds(phase):
        reference = initial = None
        for policy in POLICIES:
            rows, init_hash = verify_arm(root, phase, seed, policy)
            if reference is None:
                reference, initial = rows, init_hash
            if init_hash != initial:
                raise ValueError('Initial branch weights differ between teacher policies')
            changes = compare_rows(reference, rows)
            if policy != 'old' and changes == 0:
                raise ValueError('Teacher intervention changed no offline labels')
            folder = root / phase / 'train' / f'seed_{seed}' / policy
            for name in ('cotracker_three_final.pth', 'initialization-report.json', 'paired-steps.jsonl', 'run-config.json', 'trainer-adapter.json'):
                path = folder / name
                artifacts[path.relative_to(root).as_posix()] = sha(path)
            proofs.append({'seed': seed, 'policy': policy, 'steps': len(rows), 'frozen_backbone_exact': True,
                           'offline_label_changed_steps_vs_old': changes,
                           'median_step_seconds': statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2)})
            for slot in range(3):
                subset = [r for r in rows.values() if r['primary_index'] == slot]
                if subset:
                    costs.append({'Phase': phase, 'Seed': seed, 'Policy': policy, 'TeacherSlot': slot, 'Steps': len(subset),
                                  'StepSecondsMean': statistics.mean(r['step_seconds'] for r in subset),
                                  'TeacherSecondsMean': statistics.mean(r['label_audit']['teacher_seconds'] for r in subset),
                                  **{k: statistics.mean(r['label_audit'][k] for r in subset) for k in
                                     ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction')}})
    save_json(root / phase / 'training-audit.json', {'complete': True, 'phase': phase, 'proofs': proofs,
              'manifest_sha256': sha(root / 'manifest.json'), 'artifacts_sha256': artifacts})
    write_csv(root / phase / 'label-runtime-diagnostics.csv', costs)
    print('PAIRED TRAINING AUDIT PASSED ' + phase, flush=True)


def checked_training(root, phase):
    proof = read(root / phase / 'training-audit.json')
    if (not proof['complete'] or proof['phase'] != phase or proof['manifest_sha256'] != sha(root / 'manifest.json') or
        {(p['seed'], p['policy']) for p in proof['proofs']} != {(s, p) for s in phase_seeds(phase) for p in POLICIES} or
        len(proof['proofs']) != len(phase_seeds(phase)) * len(POLICIES) or
        any(not p['frozen_backbone_exact'] or p['steps'] != (SMOKE_STEPS if phase == 'smoke' else 1000) for p in proof['proofs'])):
        raise ValueError('Incomplete or mismatched training audit')
    checked_hashes(root, proof['artifacts_sha256'])
    expected_artifacts = {f'{phase}/train/seed_{s}/{p}/{name}' for s in phase_seeds(phase) for p in POLICIES
                          for name in ('cotracker_three_final.pth', 'initialization-report.json',
                                       'paired-steps.jsonl', 'run-config.json', 'trainer-adapter.json')}
    if set(proof['artifacts_sha256']) != expected_artifacts:
        raise ValueError('Training audit artifact coverage incomplete')
    return proof


def gpu_smoke(root):
    checked_training(root, 'smoke')
    # Import actual application entry so stale installed wheels are also caught.
    import model as entry
    torch.manual_seed(BASE_SEED)
    video = torch.rand(1, 60, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [29., 32., 32.], [59., 48., 48.]]], device='cuda')
    os.environ['TRACKRAD_SIDECAR_RUNTIME'] = str(root / 'runtime.json')
    proofs = []
    for policy in POLICIES:
        m = entry.resources.setup_model(str(root / 'smoke/train/seed_0' / policy / 'cotracker_three_final.pth'), device='cuda')
        d = {}
        m.updateformer.feature_sidecar.diagnostics = d
        with torch.no_grad():
            output = m(video=video, queries=queries, iters=2)
        if (m.feature_sidecar_kind != KIND or m.updateformer.feature_sidecar.runtime != RUNTIME or
            not m.frozen_pretrained or d.get('sidecar_calls') != 2 or not d.get('sidecar_residual_rms_max', 0) or
            any(not bool(torch.isfinite(t).all()) for t in output[:3])):
            raise ValueError('GPU trained-student inference failed')
        proofs.append({'policy': policy, **d})
        del m
        torch.cuda.empty_cache()
    save_json(root / 'smoke/gpu-inference.json', {'complete': True, 'policies': list(POLICIES), 'proofs': proofs,
              'training_audit_sha256': sha(root / 'smoke/training-audit.json')})
    print('GPU INFERENCE SMOKE PASSED: all four teachers / fixed Mamba', flush=True)


def smoke_gate(root):
    checked_training(root, 'smoke')
    proof = read(root / 'smoke/gpu-inference.json')
    if not proof['complete'] or proof['policies'] != list(POLICIES) or proof['training_audit_sha256'] != sha(root / 'smoke/training-audit.json'):
        raise ValueError('Run and pass -Stage smoke before full training')
    print('SMOKE GATE PASSED', flush=True)


def report(root, split):
    checked_training(root, 'full')
    manifest = read(root / 'manifest.json')
    folder = root / 'full' / split
    expected_cases = set(manifest['case_ids'][split])
    records, pairs, case_pairs, collected, hashes = [], [], [], {}, {}

    def load(relative, kind):
        path = folder / relative
        if read(path / 'frozen-images.json') != read(root / 'reference-images.json'):
            raise ValueError('Evaluation image changed')
        values, cases, digest, _ = load_arm(path, kind, len(expected_cases), PARAMETERS if kind != 'none' else 25385700)
        if set(cases) != expected_cases:
            raise ValueError('Official case IDs mismatch')
        hashes[relative + '/' + PROFILE + '/metrics.json'] = digest
        for case in cases:
            job = path / PROFILE / 'checkpoint/jobs' / case
            diag = read(job / 'diagnostics.json')
            if diag['config'] != manifest['student_config'] or (kind != 'none' and diag['model']['sidecar_runtime'] != RUNTIME):
                raise ValueError('Student inference settings changed')
            for artifact in ('diagnostics.json', '.complete', 'output/images/mri-linac-series-targets/output.mha'):
                p = job / artifact
                hashes[p.relative_to(folder).as_posix()] = sha(p)
        return values, cases

    ref = load('pretrained', 'none')
    records.append({'Split': split, 'Seed': -1, 'Policy': 'pretrained', **ref[0]})
    for seed in SEEDS:
        arms = {'pretrained': ref}
        for policy in POLICIES:
            a, cases = load(f'seed_{seed}/{policy}', KIND)
            arms[policy] = a, cases
            collected.setdefault(policy, []).append(a)
            records.append({'Split': split, 'Seed': seed, 'Policy': policy, **a})
        for candidate, reference in PAIRS:
            a, ac = arms[reference]
            b, bc = arms[candidate]
            pairs.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                          **{'Delta_' + k: b[k] - a[k] for k in FIELDS}})
            for case in sorted(expected_cases):
                case_pairs.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference, 'Case': case,
                                   **{'Delta_' + k: float(bc[case][v]) - float(ac[case][v]) for k, v in FIELDS.items()}})
    means, paired_means = [], []
    for policy, samples in [('pretrained', [ref[0]]), *collected.items()]:
        row = {'Split': split, 'Policy': policy, 'N': len(samples), 'Kind': 'none' if policy == 'pretrained' else KIND,
               'Parameters': 25385700 if policy == 'pretrained' else PARAMETERS, 'StudentProfile': PROFILE}
        for k in (*FIELDS, 'TimeSec'):
            row[k + '_Mean'] = statistics.mean(r[k] for r in samples)
            row[k + '_SD'] = statistics.stdev(r[k] for r in samples) if len(samples) > 1 else 0.
        means.append(row)
    for candidate, reference in PAIRS:
        subset = [r for r in pairs if (r['Candidate'], r['Reference']) == (candidate, reference)]
        row = {'Split': split, 'Candidate': candidate, 'Reference': reference, 'N': len(subset)}
        for k in FIELDS:
            values = [r['Delta_' + k] for r in subset]
            row['Delta_' + k + '_Mean'] = statistics.mean(values)
            row['Delta_' + k + '_SD'] = statistics.stdev(values)
            row['ImprovedSeeds_' + k] = sum(v > 0 if k in ('DSC', 'D98') else v < 0 for v in values)
        paired_means.append(row)
    for name, data in (('results', records), ('summary', means), ('paired-deltas', pairs), ('paired-summary', paired_means), ('paired-cases', case_pairs)):
        write_csv(folder / ('frozen-sidecar-teachers-' + name + '.csv'), data)
    save_json(folder / 'audit.json', {'complete': True, 'split': split, 'cases': sorted(expected_cases),
              'manifest_sha256': sha(root / 'manifest.json'), 'artifacts_sha256': hashes,
              'note': 'Official metrics for the named split; N is seed count, not case count. No test-based selection. '
                      'Teacher label confidence is not correspondence accuracy. Historical reference-scan runs are context only.'})
    print('OFFICIAL REPORT COMPLETE ' + split, flush=True)
    for row in means:
        print(f"{row['Policy']}: DSC={row['DSC_Mean']:.6f} +/- {row['DSC_SD']:.6f}; D98={row['D98_Mean']:.6f}", flush=True)


def test_gate(root):
    checked_training(root, 'full')
    folder = root / 'full/validation-10'
    proof = read(folder / 'audit.json')
    manifest = read(root / 'manifest.json')
    if not proof['complete'] or proof['split'] != 'validation-10' or proof['cases'] != sorted(manifest['case_ids']['validation-10']) or proof['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('Complete validation before the 38-case report')
    checked_hashes(folder, proof['artifacts_sha256'])
    frozen(root / 'test-shortlist.json', {'policies': list(POLICIES), 'seeds': list(SEEDS),
           'validation_audit_sha256': sha(folder / 'audit.json'), 'rule': 'All predeclared arms, irrespective of validation ranking.'})
    print('VALIDATION VERIFIED; all four frozen arms proceed to test-38', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'runtime', 'verify-arm', 'audit', 'verify-training', 'gpu-smoke', 'smoke-gate', 'report', 'test-gate'))
    p.add_argument('--root', type=Path, default=Path('/study'))
    p.add_argument('--prior', type=Path, default=Path('/prior'))
    p.add_argument('--package', type=Path, default=Path('/package'))
    p.add_argument('--weights', type=Path, default=Path('/checkpoints'))
    p.add_argument('--datasets', type=Path, default=Path('/datasets'))
    p.add_argument('--phase', choices=('smoke', 'full'), default='full')
    p.add_argument('--kind', choices=('Training', 'Inference'), default='Training')
    p.add_argument('--seed', type=int, choices=SEEDS, default=0)
    p.add_argument('--policy', choices=POLICIES, default='old')
    p.add_argument('--split', choices=('validation-10', 'test-38'), default='validation-10')
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.command == 'prepare': prepare(a.root, a.prior, a.package, a.weights, a.datasets)
    elif a.command == 'runtime': runtime(a.root, a.kind)
    elif a.command == 'verify-arm': verify_arm(a.root, a.phase, a.seed, a.policy)
    elif a.command == 'audit': audit(a.root, a.phase)
    elif a.command == 'verify-training': checked_training(a.root, a.phase)
    elif a.command == 'gpu-smoke': gpu_smoke(a.root)
    elif a.command == 'smoke-gate': smoke_gate(a.root)
    elif a.command == 'test-gate': test_gate(a.root)
    else: report(a.root, a.split)
