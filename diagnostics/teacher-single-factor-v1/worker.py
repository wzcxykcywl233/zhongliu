"""Resumable teacher inference / official-metric reporting; never trains a student."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys
import time
import uuid
from common import (PACKAGE, SPLITS, MASK, METRICS, read, sha, save, csv_save, graph,
                    group_profiles, protocol_files, inventory, package_hashes, verify_job,
                    commit_job, scalar, official)

NUMERICAL_ROOT = Path('/opt/app')
JOB_OUTPUTS = [MASK, 'tracks.npz', 'diagnostics.json', 'prediction.json']


def manifest_path(args):
    return args.output / 'manifest.json'


def plan(args):
    protocol = read(args.protocol / 'frozen-run.json')
    frozen_source = read(args.study / 'frozen-study.json')
    if frozen_source['OldProtocolSHA256'].lower() != sha(args.protocol / 'frozen-run.json'):
        raise ValueError('Source study belongs to another original protocol')
    for image in (args.inference_image, args.evaluation_image):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Immutable runtime image ID required')
    if args.evaluation_image != frozen_source['Images']['Evaluation']:
        raise ValueError('Official evaluator changed from the source study')
    expected = protocol_files(protocol)
    inputs = {}
    for split, folder in args.datasets.items():
        if inventory(folder) != expected[split]:
            raise ValueError('Dataset changed from frozen 40/10/38 protocol: ' + split)
        inputs.update({f'{split}/{k}': v for k, v in expected[split].items()})
    weights = {row['Name']: row['SHA256'].lower() for row in protocol['Weights']}
    for name in ('baseline_offline.pth', 'scaled_offline.pth'):
        if name not in weights or sha(args.weights / name) != weights[name]:
            raise ValueError('Unverified teacher checkpoint: ' + name)
    source = frozen_source['Source']
    numerical = {}
    for row in source:
        relative = row['File'].replace('\\', '/')
        if relative.startswith('cotracker-algorithm/') and relative.endswith('.py'):
            relative = relative[len('cotracker-algorithm/'):]
            # Reuse images copy numerical code, not newly added host tests.
            # Verify the transitive inference implementation, not absent test files.
            if not (relative.startswith('resources/') or relative.startswith('ext/co-tracker/cotracker/')
                    or relative in ('model.py', 'tuning.py')):
                continue
            path = NUMERICAL_ROOT / relative
            if sha(path) != row['SHA256'].lower():
                raise ValueError('Pinned inference source changed: ' + relative)
            numerical[relative] = row['SHA256'].lower()
    if 'resources/model.py' not in numerical or not any('cotracker3_offline.py' in p for p in numerical):
        raise ValueError('Numerical source inventory missing')
    spec, profiles, resolved = graph()
    manifest = {'schema': 1, 'protocol_sha256': sha(args.protocol / 'frozen-run.json'),
                'study_sha256': sha(args.study / 'frozen-study.json'), 'inputs': inputs,
                'weights': {k: weights[k] for k in ('baseline_offline.pth', 'scaled_offline.pth')},
                'numerical_source': numerical, 'package': package_hashes(),
                'inference_image': args.inference_image, 'evaluation_image': args.evaluation_image,
                'spec': spec, 'resolved': resolved, 'training': False, 'student_changes': False}
    path = manifest_path(args)
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Inputs/code/images changed; preserve results and use a NEW OutputRoot')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Unrecorded output directory; use a NEW OutputRoot')
        save(path, manifest)
    rows = []
    for name, profile in profiles.items():
        rows.append({'Profile': name, 'Group': profile['group'], 'Parent': profile['parent'],
                     'Factor': next(iter(profile['change']), 'reference'), 'Change': profile['change'],
                     'Config': resolved[name]})
    save(args.output / 'plan.json', {'complete': True, 'profiles': rows, 'cases': SPLITS,
                                    'query_scope': spec['query_scope'], 'target_points': spec['target_points'],
                                    'note': 'Teacher label-generator pilot, not student performance.'})
    print('PLAN VERIFIED: 40/10/38 disjoint, 32 profiles, no Mamba/training', flush=True)


def verify(args, split=None, profile=None):
    manifest = read(manifest_path(args))
    if package_hashes() != manifest['package']:
        raise ValueError('Experiment package changed; use a NEW OutputRoot')
    if sha(args.protocol / 'frozen-run.json') != manifest['protocol_sha256'] or sha(args.study / 'frozen-study.json') != manifest['study_sha256']:
        raise ValueError('Frozen source protocol changed')
    for relative, digest in manifest['numerical_source'].items():
        if sha(NUMERICAL_ROOT / relative) != digest:
            raise ValueError('Runtime numerical source changed: ' + relative)
    if split:
        expected = {k.split('/', 1)[1]: v for k, v in manifest['inputs'].items() if k.startswith(split + '/')}
        if inventory(args.datasets[split]) != expected:
            raise ValueError('Dataset changed: ' + split)
    if profile:
        checkpoint = manifest['resolved'][profile]['checkpoint']
        if sha(args.weights / checkpoint) != manifest['weights'][checkpoint]:
            raise ValueError('Teacher checkpoint changed')
    return manifest


def cases(args, split):
    manifest = read(manifest_path(args))
    return sorted({key.split('/')[1] for key in manifest['inputs'] if key.startswith(split + '/')})


def token(args, split, profile):
    return hashlib.sha256((sha(manifest_path(args)) + '/' + split + '/' + profile).encode()).hexdigest()


def job_folders(args, split, profile):
    root = args.output / split / profile
    folders = [root / 'jobs' / case for case in cases(args, split)]
    for folder in folders:
        if not verify_job(folder, token(args, split, profile), JOB_OUTPUTS):
            raise ValueError('Incomplete teacher prediction: ' + str(folder))
    return root, folders


def evaluation_token(args, split, profile):
    root, jobs = job_folders(args, split, profile)
    return hashlib.sha256((token(args, split, profile) + ''.join(sha(p / 'verified.json') for p in jobs)).encode()).hexdigest()


def prediction(case, dataset, started, completed):
    metadata = [('frame-rate', 'frame-rate.json', 'number'),
                ('magnetic-field-strength', 'b-field-strength.json', 'number'),
                ('scanned-region', 'scanned-region.json', 'string')]
    inputs = [{'interface': {'slug': slug}, 'value': scalar(dataset / case / leaf, kind)}
              for slug, leaf, kind in metadata]
    inputs += [{'interface': {'slug': 'mri-linac-series'}, 'image': {'name': case + '_frames.mha'}},
               {'interface': {'slug': 'mri-linac-target'}, 'image': {'name': 'target.mha'}}]
    return {'pk': 'jobs/' + case, 'inputs': inputs, 'status': 'Succeeded',
            'outputs': [{'interface': {'slug': 'mri-linac-series-targets',
                                      'relative_path': 'images/mri-linac-series-targets'},
                         'image': {'name': 'output.mha'}}],
            'started_at': started, 'completed_at': completed}


def input_case(dataset, case):
    import numpy as np
    import SimpleITK as sitk
    from predictor import canonical
    # No future-frame labels are opened here. They are read only by the evaluator.
    image = sitk.ReadImage(str(dataset / case / 'images' / (case + '_frames.mha')))
    video = canonical(sitk.GetArrayFromImage(image))
    first = canonical(sitk.GetArrayFromImage(sitk.ReadImage(
        str(dataset / case / 'targets' / (case + '_first_label.mha')))))
    if len(first) != 1 or first.shape[1:] != video.shape[1:] or len(video) < 2:
        raise ValueError('MRI/first-query-mask geometry mismatch')
    return image, video, first[0] > .5


def smoke(args):
    import torch
    import numpy as np
    from predictor import activate, load_teacher, track, preprocess, query_points
    activate()
    manifest = verify(args, 'validation-10')
    folder = args.output / 'smoke'
    if verify_job(folder, sha(manifest_path(args))):
        print('SMOKE VERIFIED CACHE', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    case = cases(args, 'validation-10')[0]
    _, array, first = input_case(args.datasets['validation-10'], case)
    queries = query_points(first, 32)
    # Three span-10 levels exercise later memory reads, not just initial queries.
    smoke_frames = min(23, len(array))
    if smoke_frames < 21:
        raise ValueError('Smoke case must have at least 21 frames to exercise query-memory reads')
    video = preprocess(array[:smoke_frames], 'raw')
    proofs = []
    for name in ('old', 'scaled', 'diverse', 'anchor_rollback'):
        config = manifest['resolved'][name]
        model = load_teacher(args.weights / config['checkpoint'])
        outputs, diagnostics = track(model, video, queries, config, 123)
        repeated, _ = track(model, video, queries, config, 123)
        drift = [float((a - b).abs().max()) for a, b in zip(outputs, repeated)]
        if any(d > 1e-3 for d in drift):
            raise ValueError('Smoke repeat changed beyond numerical tolerance: ' + name)
        if not np.allclose(outputs[0][:, 0], queries[..., 1:], atol=1e-3, rtol=0):
            raise ValueError('Teacher failed first-frame point identity: ' + name)
        proofs.append({'profile': name, 'repeat_max_abs_PVC': drift, 'diagnostics': diagnostics,
                       'frozen_eval': not model.training and not any(p.requires_grad for p in model.parameters())})
        del model
        torch.cuda.empty_cache()
        print('GPU SMOKE PASSED ' + name, flush=True)
    verify(args, 'validation-10')
    save(folder / 'summary.json', {'complete': True, 'frames': smoke_frames, 'points': 32, 'proofs': proofs})
    commit_job(folder, sha(manifest_path(args)), ['summary.json'])


def infer(args):
    import numpy as np
    import torch
    import SimpleITK as sitk
    from predictor import load_teacher, track, preprocess, query_points, label_diagnostics, render, tensor_sha
    manifest = verify(args, args.split, args.profile)
    if not verify_job(args.output / 'smoke', sha(manifest_path(args))):
        raise ValueError('Run GPU smoke before full-case teacher experiments')
    if args.split == 'test-38':
        verify_selection(args)
        if args.profile not in read(args.output / 'selection.json')['test_profiles']:
            raise ValueError('Profile was not selected before test')
    config = manifest['resolved'][args.profile]
    root = args.output / args.split / args.profile
    root.mkdir(parents=True, exist_ok=True)
    model = None
    for case in cases(args, args.split):
        final = root / 'jobs' / case
        if verify_job(final, token(args, args.split, args.profile), JOB_OUTPUTS):
            print('CACHE VERIFIED ' + args.profile + '/' + case, flush=True)
            continue
        if model is None:
            torch.set_num_threads(1)
            model = load_teacher(args.weights / config['checkpoint'])
        image, array, first = input_case(args.datasets[args.split], case)
        queries = query_points(first, manifest['spec']['target_points'])
        extra = query_points(first, config['extra_border_points']) if config['extra_border_points'] else None
        video = preprocess(array, config['intensity'])
        case_seed = manifest['spec']['seed'] + int(hashlib.sha256(case.encode()).hexdigest()[:6], 16)
        attempt = root / '.attempts' / (case + '-' + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        started = datetime.now(timezone.utc).isoformat()
        torch.cuda.synchronize()
        tic = time.perf_counter()
        outputs, mechanism = track(model, video, queries, config, case_seed, extra)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tic
        mask = render(outputs, array.shape[1:], first)
        completed = datetime.now(timezone.utc).isoformat()
        output_path = attempt / MASK
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8))
        output.CopyInformation(image)
        sitk.WriteImage(output, str(output_path), True)
        check = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
        if not np.array_equal(check, mask.transpose(2, 1, 0)):
            raise ValueError('MHA output roundtrip failed')
        np.savez_compressed(attempt / 'tracks.npz', queries=queries.numpy(),
                            trajectories=outputs[0].numpy(), visibility=outputs[1].numpy(), confidence=outputs[2].numpy())
        report = {'profile': args.profile, 'case': case, 'split': args.split, 'config': config,
                  'checkpoint_sha256': manifest['weights'][config['checkpoint']], 'seconds_tracking': seconds,
                  'target_query_sha256': tensor_sha(queries), 'mechanism': mechanism,
                  'labels': label_diagnostics(outputs), 'student_trained': False}
        save(attempt / 'diagnostics.json', report)
        save(attempt / 'prediction.json', prediction(case, args.datasets[args.split], started, completed))
        # Verify the actual case inputs and checkpoint again immediately before commit.
        for key, digest in manifest['inputs'].items():
            if key.startswith(args.split + '/' + case + '/') and sha(args.datasets[args.split] / key.split('/', 1)[1]) != digest:
                raise ValueError('Case inputs changed during inference')
        if sha(args.weights / config['checkpoint']) != manifest['weights'][config['checkpoint']]:
            raise ValueError('Teacher weights changed during inference')
        commit_job(attempt, token(args, args.split, args.profile), JOB_OUTPUTS)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            final.rename(root / '.attempts' / (case + '-unverified-' + uuid.uuid4().hex[:8]))
        attempt.rename(final)
        print(f'COMMITTED {args.profile}/{case} tracking={seconds:.2f}s', flush=True)
    prepare_evaluation(args)


def prepare_evaluation(args):
    root, folders = job_folders(args, args.split, args.profile)
    predictions = [read(p / 'prediction.json') for p in folders]
    save(root / 'predictions.json', predictions)
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    hit = verify_job(evaluation, evaluation_token(args, args.split, args.profile), ['metrics.json'])
    save(root / 'evaluation-ready.json', {'needs_evaluation': not hit})


def record(args):
    verify(args, args.split, args.profile)
    root, _ = job_folders(args, args.split, args.profile)
    official(root / 'evaluation/metrics.json', cases(args, args.split))
    commit_job(root / 'evaluation', evaluation_token(args, args.split, args.profile), ['metrics.json'])
    print('OFFICIAL TEACHER METRICS VERIFIED ' + args.split + '/' + args.profile, flush=True)


def metrics_for(args, split, name):
    root = args.output / split / name
    if not verify_job(root / 'evaluation', evaluation_token(args, split, name), ['metrics.json']):
        raise ValueError('Official metrics incomplete: ' + split + '/' + name)
    return official(root / 'evaluation/metrics.json', cases(args, split))


def selection_value(args):
    names = group_profiles(args.group)
    _, profiles, _ = graph()
    metrics = {n: metrics_for(args, 'validation-10', n)[0] for n in names}
    baseline = metrics['old']
    # Frozen deterministic rule; no test metrics are accessed here.
    candidates = [n for n in names if n not in ('old', 'scaled') and metrics[n]['DSC'] > baseline['DSC']]
    # scaled is itself the first teacher-weight intervention and must be eligible.
    if 'scaled' in names and metrics['scaled']['DSC'] > baseline['DSC']:
        candidates.append('scaled')
    candidates.sort(key=lambda n: (-metrics[n]['DSC'], metrics[n]['HD95'], metrics[n]['CD'], -metrics[n]['D98'], n))
    winners = candidates[:3]
    needed = {'old', 'scaled'}
    for n in winners:
        needed.add(n)
        if profiles[n]['parent']:
            needed.add(profiles[n]['parent'])
    tested = [n for n in profiles if n in needed]
    return {'schema': 1, 'manifest_sha256': sha(manifest_path(args)), 'selection_split': 'validation-10',
            'group': args.group, 'eligible_profiles': names, 'winners': winners, 'test_profiles': tested,
            'rule': 'Top 3 DSC above old teacher; ties HD95,CD,D98,name. Always old/scaled and direct parents. No test selection.',
            'validation_metrics_sha256': {n: sha(args.output / 'validation-10' / n / 'evaluation/metrics.json') for n in names},
            'note': 'DSC ranking is not an all-metric win. D98 and label diagnostics remain mandatory tradeoffs.'}


def select(args):
    value = selection_value(args)
    path = args.output / 'selection.json'
    if path.exists() and read(path) != value:
        raise ValueError('Selection already frozen; use a new study, do not retune using test')
    save(path, value)
    print('FROZEN VALIDATION SHORTLIST ' + ','.join(value['test_profiles']), flush=True)


def verify_selection(args):
    selected = read(args.output / 'selection.json')
    if selected['manifest_sha256'] != sha(manifest_path(args)) or selected['selection_split'] != 'validation-10':
        raise ValueError('Invalid frozen selection')
    old_group = args.group
    args.group = selected['group']
    try:
        if selection_value(args) != selected:
            raise ValueError('Frozen validation selection changed')
    finally:
        args.group = old_group


def report(args):
    _, profiles, configs = graph()
    if args.split == 'test-38':
        verify_selection(args)
        names = read(args.output / 'selection.json')['test_profiles']
    else:
        names = group_profiles(args.group)
    metrics = {n: metrics_for(args, args.split, n) for n in names}
    rows, deltas, pairs, label_rows = [], [], [], []
    for name in names:
        values, case_metrics = metrics[name]
        parent = profiles[name]['parent']
        rows.append({'Split': args.split, 'ReportKind': 'teacher-label-generator', 'Profile': name,
                     'Parent': parent, 'Factor': next(iter(profiles[name]['change']), 'reference'), **values})
        references = list(dict.fromkeys(n for n in (parent, 'old', 'scaled') if n and n != name and n in metrics))
        for ref in references:
            other, per_case = metrics[ref]
            deltas.append({'Split': args.split, 'Profile': name, 'Reference': ref,
                           **{'Delta_' + k: values[k] - other[k] for k in METRICS},
                           'RuntimeRatio': values['TimeSec'] / other['TimeSec'] if other['TimeSec'] > 0 else None})
            for case in cases(args, args.split):
                pairs.append({'Split': args.split, 'Profile': name, 'Reference': ref, 'Case': case,
                              **{'Delta_' + k: case_metrics[case][v] - per_case[case][v] for k, v in METRICS.items()}})
        for case in cases(args, args.split):
            folder = args.output / args.split / name / 'jobs' / case
            diagnostic = read(folder / 'diagnostics.json')
            row = {'Profile': name, 'Case': case, 'Parent': parent, **diagnostic['labels'],
                   'SecondsTracking': diagnostic['seconds_tracking']}
            if parent in metrics:
                import numpy as np
                with np.load(folder / 'tracks.npz', allow_pickle=False) as a, np.load(
                        args.output / args.split / parent / 'jobs' / case / 'tracks.npz', allow_pickle=False) as b:
                    if not np.array_equal(a['queries'], b['queries']):
                        raise ValueError('Parent/child target point identities changed')
                    row.update(MeanPositionDeltaPixels=float(np.linalg.norm(a['trajectories'][:, 1:] - b['trajectories'][:, 1:], axis=-1).mean()),
                               MeanAbsVisibilityDelta=float(np.abs(a['visibility'][:, 1:] - b['visibility'][:, 1:]).mean()),
                               MeanAbsConfidenceDelta=float(np.abs(a['confidence'][:, 1:] - b['confidence'][:, 1:]).mean()))
            label_rows.append(row)
    suffix = args.split + (('-' + args.group) if args.split == 'validation-10' else '')
    for leaf, data in (('results', rows), ('deltas', deltas), ('paired-cases', pairs), ('label-diagnostics', label_rows)):
        csv_save(args.output / f'teacher-{suffix}-{leaf}.csv', data)
    save(args.output / f'teacher-{suffix}-summary.json', {'complete': True, 'split': args.split,
         'cases': len(cases(args, args.split)), 'profiles': names, 'training': False,
         'official_metrics_copied': True, 'target_points': 250,
         'note': 'Teacher generator quality, NOT trained student accuracy. No point-GT or blind-test claim. First-frame boundary queries only.'})
    print('TEACHER REPORT COMPLETE ' + suffix, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['plan', 'smoke', 'infer', 'record', 'select', 'report'], required=True)
    for name in ('protocol', 'study', 'weights', 'output', 'train', 'validation', 'test'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=['validation-10', 'test-38'], default='validation-10')
    parser.add_argument('--profile', default='old')
    parser.add_argument('--group', default='weights')
    parser.add_argument('--inference-image', default='')
    parser.add_argument('--evaluation-image', default='')
    args = parser.parse_args()
    args.datasets = {'train-40': args.train, 'validation-10': args.validation, 'test-38': args.test}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action != 'plan':
        verify(args)
    globals()[args.action](args)


if __name__ == '__main__':
    main()
