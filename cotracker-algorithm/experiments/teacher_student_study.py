"""Checked training, paired label diagnostics and official student split reports."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext/co-tracker')]
from cotracker.utils.offline_teacher_policy import POLICIES, SETTINGS, offline_teacher_forward
from cotracker.utils.teacher_sampling import TeacherPairSampler
from cotracker.models.core.cotracker.backbone_growth import save_json
from audit_backbone_growth import file_sha, validate_checkpoint
from summarize_backbone_growth import FIELDS, PROFILE, load_arm, write_csv

BASE_SEED = 20261017
LABEL_KEYS = ('tracks_sha256', 'visibility_sha256', 'confidence_sha256')
INPUT_KEYS = ('case_names', 'queries_sha256', 'video_sha256', 'primary_index', 'auxiliary_index')


def read_records(path, steps, policy):
    rows = {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        row = json.loads(line)
        i = row['step']
        if type(i) is not int or not 0 <= i < steps or row['teacher_policy'] != policy or row['architecture'] != 'base':
            raise ValueError(f'invalid audit row {path}:{number}')
        if row['auxiliary_index'] is not None or row['primary_index'] not in (0, 1, 2):
            raise ValueError('teacher pool/auxiliary intervention changed')
        label = row['label_audit']
        expected_type = ('online', 'online_cotracker_three', 'offline_cotracker_three')[row['primary_index']]
        counts = label['query_time_counts']
        if (label['teacher_type'] != expected_type or label['policy'] != policy or len(counts) != 10 or
            any(type(x) is not int or x < 0 for x in counts) or sum(counts) != 384):
            raise ValueError('wrong executed teacher or query coverage')
        for key in (*LABEL_KEYS, 'support_draw_sha256'):
            if len(label[key]) != 64 or any(c not in '0123456789abcdef' for c in label[key]):
                raise ValueError('invalid label/input checksum')
        for key in ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction'):
            if not 0 <= label[key] <= 1:
                raise ValueError('invalid pseudo-label diagnostic')
        if row['primary_index'] == 2 and (label['support_used'], label['iterations']) != SETTINGS[policy][1:]:
            raise ValueError('teacher policy was not executed')
        if not all(math.isfinite(float(x)) and x >= 0 for x in (row['step_seconds'], label['teacher_seconds'])) or not math.isfinite(row['loss']):
            raise ValueError('invalid training timing/loss')
        # Duplicate replay is allowed ONLY with identical inputs and raw labels.
        signature = lambda r: [r[k] for k in INPUT_KEYS] + [r['label_audit'][k] for k in (*LABEL_KEYS, 'support_draw_sha256', 'query_time_counts')]
        if i in rows and signature(rows[i]) != signature(row):
            raise ValueError(f'resumed step input or labels changed: {path}:{i}')
        rows[i] = row
    if set(rows) != set(range(steps)):
        raise ValueError('missing paired audit steps: ' + str(path))
    return rows


def audit(root, seeds, steps, max_median_seconds):
    proof, hashes, costs, artifacts = [], {}, [], {}
    torch.set_num_threads(1)
    for seed in seeds:
        reference, reference_config = None, None
        for policy in POLICIES:
            folder = root / 'train' / f'seed_{seed}' / policy
            config = json.loads((folder / 'run-config.json').read_text(encoding='utf-8-sig'))
            if config['steps'] != steps:
                raise ValueError('training step count differs from frozen run config')
            comparable = {k: v for k, v in config.items() if k not in ('policy', 'offline_checkpoint')}
            if reference_config is None:
                reference_config = comparable
            elif comparable != reference_config:
                raise ValueError('student training settings differ between arms')
            init = json.loads((folder / 'initialization-report.json').read_text())
            if init['architecture'] != 'base' or not init['initialization_outputs_exact'] or init['added_parameters']:
                raise ValueError('student initialization changed')
            final = folder / 'cotracker_three_final.pth'
            ckpt = torch.load(final, map_location='cpu', weights_only=True)
            if ckpt.get('teacher_student_policy') != policy:
                raise ValueError('final checkpoint teacher provenance missing')
            validate_checkpoint(folder, 'base', steps)
            rows = read_records(folder / 'paired-steps.jsonl', steps, policy)
            sampler = TeacherPairSampler(3, seed=BASE_SEED + seed)
            if any(rows[i]['primary_index'] != sampler.sample().primary for i in range(steps)):
                raise ValueError('executed teacher schedule differs from the declared seed')
            if reference is None:
                reference = rows
            else:
                for i, row in rows.items():
                    ref = reference[i]
                    if any(ref[k] != row[k] for k in INPUT_KEYS) or any(
                        ref['label_audit'][k] != row['label_audit'][k] for k in ('support_draw_sha256', 'query_time_counts')):
                        raise ValueError(f'teacher/case/query/video/support pairing failed: {policy}:{i}')
                    if row['primary_index'] != 2 and any(ref['label_audit'][k] != row['label_audit'][k] for k in LABEL_KEYS):
                        raise ValueError('unmodified teacher labels changed')
            offline = [r for r in rows.values() if r['primary_index'] == 2]
            if not offline:
                raise ValueError('offline teacher never sampled; smoke/pilot cannot establish the intervention')
            durations = [r['step_seconds'] for i, r in rows.items() if i >= min(2, steps - 1)]
            median = statistics.median(durations)
            if median > max_median_seconds:
                raise ValueError(f'training too slow: {policy} median={median:.2f}s; no larger run allowed')
            hashes[str(final.relative_to(root))] = file_sha(final)
            for name in ('run-config.json', 'paired-steps.jsonl', 'initialization-report.json'):
                path = folder / name
                artifacts[str(path.relative_to(root))] = file_sha(path)
            raw_changes = sum(any(r['label_audit'][k] != reference[i]['label_audit'][k] for k in LABEL_KEYS)
                              for i, r in rows.items() if r['primary_index'] == 2)
            proof.append({'seed': seed, 'policy': policy, 'paired_steps': steps, 'offline_steps': len(offline),
                          'offline_label_changed_steps_vs_old': raw_changes, 'median_step_seconds': median})
            for slot in (0, 1, 2):
                subset = [r for r in rows.values() if r['primary_index'] == slot]
                if subset:
                    costs.append({'Seed': seed, 'Policy': policy, 'TeacherSlot': slot, 'Steps': len(subset),
                                  'TeacherSecondsMean': statistics.mean(r['label_audit']['teacher_seconds'] for r in subset),
                                  'StepSecondsMean': statistics.mean(r['step_seconds'] for r in subset),
                                  **{k: statistics.mean(r['label_audit'][k] for r in subset) for k in
                                     ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction')}})
    result = {'complete': True, 'seeds': seeds, 'steps': steps, 'policies': list(POLICIES), 'proofs': proof,
              'final_checkpoints_sha256': hashes, 'paired_inputs_and_unchanged_teacher_labels': True,
              'training_artifacts_sha256': artifacts,
              'note': 'Raw teacher V/C and smoothness are not calibrated correspondence accuracy. Timing excludes data loading/checkpoint I/O.'}
    write_csv(root / 'label-runtime-diagnostics.csv', costs)
    save_json(root / 'training-audit.json', result)
    print(json.dumps(result), flush=True)


def verify_training(root, seeds, steps):
    audit = json.loads((root / 'training-audit.json').read_text())
    if not audit['complete'] or audit['seeds'] != seeds or audit['steps'] != steps or audit['policies'] != list(POLICIES):
        raise ValueError('training audit does not match requested evaluation')
    for relative, expected in audit['final_checkpoints_sha256'].items():
        if file_sha(root / relative) != expected:
            raise ValueError('trained checkpoint changed')
    for relative, expected in audit['training_artifacts_sha256'].items():
        if file_sha(root / relative) != expected:
            raise ValueError('training audit artifact changed')
    return audit


def report(root, seeds, steps, split):
    verify_training(root, seeds, steps)
    count = {'validation-10': 10, 'test-38': 38}[split]
    folder = root / split
    pretrained, cases, digest = load_arm(folder / 'pretrained', 'base', count)
    records = [{'Split': split, 'Seed': -1, 'Policy': 'pretrained', **pretrained}]
    hashes = {f'pretrained/{PROFILE}/metrics.json': digest}
    paired, per_case, values = [], [], {}
    for seed in seeds:
        arms = {'pretrained': (pretrained, cases)}
        for policy in POLICIES:
            arm = folder / f'seed_{seed}' / policy
            aggregate, rows, digest = load_arm(arm, 'base', count)
            if set(rows) != set(cases):
                raise ValueError('mismatched case IDs')
            arms[policy] = aggregate, rows
            values.setdefault(policy, []).append(aggregate)
            hashes[str((arm / PROFILE / 'metrics.json').relative_to(folder))] = digest
            records.append({'Split': split, 'Seed': seed, 'Policy': policy, **aggregate})
        contrasts = [('scaled', 'old'), ('grid0', 'scaled'), ('iter6', 'grid0')]
        contrasts += [(p, 'pretrained') for p in POLICIES] + [(p, 'old') for p in POLICIES[2:]]
        for candidate, reference in contrasts:
            a, ac = arms[reference]
            b, bc = arms[candidate]
            paired.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                           **{f'Delta_{k}': b[k] - a[k] for k in FIELDS}})
            for case in sorted(cases):
                per_case.append({'Split': split, 'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                                 'Case': case, **{f'Delta_{k}': bc[case][v] - ac[case][v] for k, v in FIELDS.items()}})
    means = []
    for policy, rows in [('pretrained', [pretrained]), *values.items()]:
        row = {'Split': split, 'Policy': policy, 'N': len(rows), 'Parameters': 25385700}
        for key in (*FIELDS, 'TimeSec'):
            row[key + '_Mean'] = statistics.mean(r[key] for r in rows)
            row[key + '_SD'] = statistics.stdev(r[key] for r in rows) if len(rows) > 1 else 0.
        means.append(row)
    for name, rows in [('results', records), ('summary', means), ('paired-deltas', paired), ('paired-cases', per_case)]:
        write_csv(folder / f'teacher-student-{name}.csv', rows)
    paired_means = []
    for candidate, reference in dict.fromkeys((r['Candidate'], r['Reference']) for r in paired):
        subset = [r for r in paired if (r['Candidate'], r['Reference']) == (candidate, reference)]
        row = {'Split': split, 'Candidate': candidate, 'Reference': reference, 'N': len(subset)}
        for key in FIELDS:
            samples = [r[f'Delta_{key}'] for r in subset]
            row[f'Delta_{key}_Mean'] = statistics.mean(samples)
            row[f'Delta_{key}_SD'] = statistics.stdev(samples) if len(samples) > 1 else 0.
        paired_means.append(row)
    write_csv(folder / 'teacher-student-paired-summary.csv', paired_means)
    result = {'complete': True, 'split': split, 'cases': count, 'seeds': seeds, 'steps': steps,
              'metrics_sha256': hashes, 'inference_profile': PROFILE, 'policies': list(POLICIES),
              'note': 'Official STUDENT metrics, not teacher-generator scores. Test-38 is repeatedly inspected, not a blind test. No p-values. All four arms predeclared; no test-based selection.'}
    save_json(folder / 'teacher-student-audit.json', result)
    print(json.dumps(means), flush=True)


def gpu_proof(root, checkpoints):
    """Actual offline model compatibility; small synthetic input is not quality evidence."""
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_uniformly_sampled_pts
    torch.manual_seed(BASE_SEED)
    video = torch.rand(1, 10, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0., 16., 16.], [2., 24., 24.], [5., 32., 32.], [9., 40., 40.]]], device='cuda')
    original = queries.clone()
    support = get_uniformly_sampled_pts(100, 10, (64, 64), device=video.device)
    proofs = []
    for policy in POLICIES:
        weight, count, iterations = SETTINGS[policy]
        model = build_cotracker(str(checkpoints / weight), offline=True, window_len=60).cuda().eval().requires_grad_(False)
        with torch.no_grad():
            combined = torch.cat((queries, support), 1) if count else queries
            direct = tuple(x[:, :, :4] for x in model(video, combined, iters=iterations)[:3])
            adapted = offline_teacher_forward(model, video, queries, support, policy)
        if not all(torch.equal(a, b) for a, b in zip(direct, adapted)) or not torch.equal(queries, original):
            raise ValueError('teacher adapter differs from direct arbitrary-time inference')
        if not all(torch.isfinite(x).all() for x in adapted):
            raise ValueError('GPU teacher output nonfinite')
        proofs.append({'policy': policy, 'direct_outputs_exact': True, 'query_times': queries[0, :, 0].cpu().tolist()})
        del model
    save_json(root / 'gpu-teacher-proof.json', {'complete': True, 'proofs': proofs})
    print('GPU TEACHER ADAPTER PROOF PASSED', flush=True)


def runtime_proof(root):
    import SimpleITK
    import torchvision
    from pytorch_lightning.lite import LightningLite
    import cotracker.utils.offline_teacher_policy as policy_module
    expected = ROOT / 'ext/co-tracker/cotracker/utils/offline_teacher_policy.py'
    if Path(policy_module.__file__).resolve() != expected.resolve():
        raise ValueError('stale installed CoTracker module shadows copied source')
    save_json(root / 'runtime-proof.json', {'complete': True, 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'teacher_policy_source': str(expected),
              'teacher_policy_sha256': file_sha(expected), 'trainer_sha256': file_sha(ROOT / 'ext/co-tracker/train_on_real_data.py')})
    print('LOCAL RUNTIME IMPORTS VERIFIED', flush=True)


def student_smoke(root):
    from resources.model import setup_model
    proofs = []
    for policy in POLICIES:
        path = root / 'train/seed_0' / policy / 'cotracker_three_final.pth'
        model = setup_model(str(path), device='cuda').eval()
        generator = torch.Generator().manual_seed(BASE_SEED)
        video = (torch.rand(1, 3, 3, 64, 64, generator=generator)*255).cuda()
        queries = torch.tensor([[[0., 16., 16.], [1., 32., 32.], [2., 48., 48.]]], device='cuda')
        with torch.no_grad():
            result = model(video, queries, iters=2)[:3]
        if result[0].shape != (1, 3, 3, 2) or not all(torch.isfinite(x).all() for x in result):
            raise ValueError('trained student failed inference smoke')
        proofs.append({'policy': policy, 'checkpoint_sha256': file_sha(path), 'finite_outputs': True})
        del model
    save_json(root / 'gpu-student-inference-proof.json', {'complete': True, 'proofs': proofs})
    print('GPU TRAINED STUDENT INFERENCE SMOKE PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['runtime-proof', 'gpu-proof', 'student-smoke', 'audit', 'verify-training', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path('/checkpoints'))
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--split', choices=['validation-10', 'test-38'], default='validation-10')
    parser.add_argument('--max-median-seconds', type=float, default=15.)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.command == 'runtime-proof':
        runtime_proof(args.root)
    elif args.command == 'gpu-proof':
        gpu_proof(args.root, args.checkpoints)
    elif args.command == 'student-smoke':
        student_smoke(args.root)
    elif args.command == 'audit':
        audit(args.root, seeds, args.steps, args.max_median_seconds)
    elif args.command == 'verify-training':
        verify_training(args.root, seeds, args.steps)
    else:
        report(args.root, seeds, args.steps, args.split)
