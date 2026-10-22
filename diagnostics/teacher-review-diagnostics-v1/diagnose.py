"""Frozen teacher-only probes. No student training or test-set access."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import time
from unittest.mock import patch

import numpy as np
import measures as M

PACKAGE = Path(__file__).resolve().parent
CHECKOUT = Path(os.environ.get('SIDECAR_CHECKOUT', '/checkout'))
APP = Path(os.environ.get('COTRACKER_APP_DIR', '/opt/app'))
RULE = {'version': 1, 'height': 384, 'width': 512, 'frames': 60, 'queries': 128,
        'synthetic_motions': ['static', 'translation', 'rotation_zoom'],
        'query_times': 'all zero; diagnostic probes, not a replay of training steps',
        'synthetic_queries': '64 first-mask contour + 64 interior grid',
        'real_queries': '128 first-mask contour', 'real_clip': 'first min(60,T) consecutive frames, no looping',
        'contour_sampler': 'largest foreground boundary pixel centers; cap count at unique pixels; not spline/arclength',
        'normalization': 'exact frozen training dataset reader: whole-volume robust 1/99 then resize',
        'geometry': 'exact frozen training dataset moveaxis(-1,0), including masks',
        'reference': 'synthetic: known image warp EPE; real: unsigned boundary distance, not point GT',
        'thresholds': [2, 4], 'no_test': True, 'no_training': True, 'no_selection': True}
NOTE = ('Teacher-only diagnostic, not student/official performance. Known image warps are not natural MRI motion. '
        'Real boundary distance is not point correspondence accuracy; boundary sliding and shared errors remain possible. '
        'Queries/clip selection differ from training; no claim of exact resumed-step replay. '
        'No p-values or patient-independent confidence intervals. No teacher/student weights are updated.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def frozen(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Frozen input/plan changed; use a NEW OutputRoot: ' + str(path))
    else:
        save(path, value)


def safe(root, relative):
    relative = str(relative).replace('\\', '/')
    if relative.startswith('/') or ':' in relative or '..' in relative.split('/'):
        raise ValueError('Unsafe relative path')
    target, root = (Path(root)/relative).resolve(), Path(root).resolve()
    if root not in target.parents:
        raise ValueError('Path escaped input')
    return target


def check(root, hashes):
    for relative, digest in hashes.items():
        if sha(safe(root, relative)) != digest.lower():
            raise ValueError('Input hash mismatch: ' + str(relative))


def load_protocol():
    folder = CHECKOUT/'diagnostics/teacher-collaboration-v1'
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('reviewed_collaboration', folder/'protocol.py')
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    import shared
    return p, shared.legacy


def job_list(cases, scope):
    train, val = cases['train-40'], cases['validation-10']
    if scope == 'smoke':
        train, val = train[:1], val[:1]
    elif scope != 'full':
        raise ValueError('Bad scope')
    return ([{'Mode': 'synthetic', 'Case': c, 'Motion': m, 'Split': 'train-40'}
             for c in train for m in RULE['synthetic_motions']] +
            [{'Mode': 'real', 'Case': c, 'Motion': 'natural', 'Split': 'validation-10'} for c in val])


def key(job):
    return job['Mode'] + '/' + job['Case'] + '/' + job['Motion']


def prepare(root, prior, collaboration, datasets, weights):
    old, original = read(collaboration/'manifest.json'), read(prior/'frozen-study.json')
    if old['prior_sha256'] != sha(prior/'frozen-study.json'):
        raise ValueError('Collaboration/prior provenance mismatch')
    check(CHECKOUT, old['source_sha256'])
    check(CHECKOUT, old['dependency_sha256'])
    check(CHECKOUT/'diagnostics/teacher-collaboration-v1', old['package_sha256'])
    check(weights, old['weights_sha256'])
    p, legacy = load_protocol()
    if old['collaboration_rule'] != p.COLLABORATION_RULE or old['pools'] != p.matrix():
        raise ValueError('Not the completed fixed collaboration study')
    runtime = legacy.verify_runtime_sources(APP, old['source_sha256'])
    if os.environ.get('DIAGNOSTIC_IMAGE_ID') != old['images']['Training']:
        raise ValueError('Wrong pinned training runtime')
    audit = read(collaboration/'audits/confirm.json')
    if not audit['complete']:
        raise ValueError('Source training audit incomplete')
    # Reuse only evidence/protocol, not cached teacher arrays: old logs store hashes, not trajectories.
    evidence = {f: sha(collaboration/f) for f in ('manifest.json', 'audits/confirm.json',
                'reports/confirm/label-runtime.csv')}
    cases, inventory = {}, {}
    seen = set()
    for split, count in [('train-40', 40), ('validation-10', 10), ('test-38', 38)]:
        rows = [r for r in original['Dataset'] if r['Split'] == split]
        ids = sorted({r['Case'] for r in rows})
        if len(ids) != count or seen.intersection(ids) or ids != old['case_ids'][split]:
            raise ValueError('Wrong or overlapping case IDs')
        seen.update(ids)
        cases[split] = ids
        if split == 'test-38':
            continue  # IDs only; never open or mount test data.
        data = datasets/('train' if split == 'train-40' else 'validation')
        hashes = {}
        for row in rows:
            parts = PureWindowsPath(row['File']).parts
            where = [i for i, name in enumerate(parts) if name == row['Case']]
            if len(where) != 1:
                raise ValueError('Ambiguous dataset path')
            relative = '/'.join(parts[where[0]:])
            if relative in hashes:
                raise ValueError('Duplicate dataset file')
            hashes[relative] = row['SHA256'].lower()
        check(data, hashes)
        if {f.relative_to(data).as_posix() for f in data.rglob('*') if f.is_file()} != set(hashes):
            raise ValueError('Dataset file inventory changed: ' + split)
        if sorted(p.name for p in data.iterdir() if p.is_dir()) != ids:
            raise ValueError('Unexpected dataset case')
        inventory[split] = hashes
    deps = {**old['source_sha256'], **old['dependency_sha256'],
            **{'diagnostics/teacher-collaboration-v1/'+n: d for n, d in old['package_sha256'].items()}}
    value = {'schema': 1, 'rule': RULE, 'note': NOTE, 'case_ids': cases, 'data_sha256': inventory,
             'weights_sha256': old['weights_sha256'], 'teacher_pool': p.B.baseline(),
             'source_sha256': deps, 'runtime_sha256': runtime, 'collaboration_sha256': evidence,
             'image': old['images']['Training'],
             'package_sha256': {f.name: sha(f) for f in PACKAGE.iterdir() if f.suffix in ('.py', '.ps1', '.txt')},
             'jobs': job_list(cases, 'full')}
    frozen(root/'manifest.json', value)
    print('PLAN VERIFIED: 120 known-warp train-40 probes + 10 real validation-10 probes; no student/test/training', flush=True)
    return p


def cache(root, job):
    folder = root/'jobs'/key(job)
    marker = folder/'verified.json'
    if not marker.exists():
        return None
    proof = read(marker)
    if proof['job'] != job or proof['manifest_sha256'] != sha(root/'manifest.json'):
        raise ValueError('Stale cache binding: ' + key(job))
    check(folder, proof['files'])
    if set(proof['files']) != {'outputs.npz', 'rows.json', 'proof.json'}:
        raise ValueError('Incomplete cached job')
    info = read(folder/'proof.json')
    if not info.get('teacher_parameters_exact') or not info.get('rules_match_original'):
        raise ValueError('Missing teacher freeze/rule proof')
    return folder


def contour(mask, n):
    from scipy.ndimage import binary_erosion, label
    labels, count = label(mask)
    if not count:
        raise ValueError('Empty first-query contour')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    region = labels == sizes.argmax()
    xy = np.argwhere(region & ~binary_erosion(region))[:, [1, 0]]
    if len(xy) < 4:
        raise ValueError('Degenerate first-query boundary')
    n = min(n, len(xy))
    return xy[np.linspace(0, len(xy), n, endpoint=False).astype(int)].astype(np.float32)


def make_input(data, job):
    import torch
    import torch.nn.functional as F
    from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset as DS
    c, h, w = job['Case'], RULE['height'], RULE['width']
    case = data/c
    video = DS._read_mha(case/'images'/f'{c}_frames.mha')
    first = DS._read_mask_mha(case/'targets'/f'{c}_first_label.mha')
    if first.shape != (1, *video.shape[1:]) or len(video) < 2:
        raise ValueError('Query mask/image geometry mismatch')
    mask = F.interpolate(first[:, None].float(), (h, w), mode='nearest')[0, 0].numpy() > .5
    native_frames = len(video)
    if job['Mode'] == 'synthetic':
        source = F.interpolate(video[:1, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        xy = contour(mask, 64)
        xx, yy = np.meshgrid(np.linspace(16, w-17, 8), np.linspace(16, h-17, 8))
        xy = np.concatenate([xy, np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)])
        transforms = M.matrices(job['Motion'], RULE['frames'], h, w)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([x, y, torch.ones_like(x)], -1).float()
        inverse = torch.from_numpy(np.linalg.inv(transforms).astype(np.float32))
        grids = torch.einsum('tij,hwj->thwi', inverse, coords)[..., :2]
        grids[..., 0] = 2*grids[..., 0]/(w-1)-1
        grids[..., 1] = 2*grids[..., 1]/(h-1)-1
        frames = F.grid_sample(source[None, None].expand(len(transforms), -1, -1, -1), grids,
                               mode='bilinear', padding_mode='zeros', align_corners=True)
        truth = M.transform_points(xy, transforms)[None]
    else:
        frames = F.interpolate(video[:RULE['frames'], None], (h, w), mode='bilinear', align_corners=False)
        xy, truth, transforms = contour(mask, 128), None, None
    queries = torch.from_numpy(np.concatenate([np.zeros((len(xy), 1), dtype=np.float32), xy], -1))[None]
    frames = frames[None].expand(-1, -1, 3, -1, -1).contiguous().float()
    return frames, queries, truth, transforms, native_frames


def real_masks(data, job, frames, native_frames):
    import torch.nn.functional as F
    from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset as DS
    case = data/job['Case']
    options = [case/'targets'/(job['Case']+suffix) for suffix in ('_staple_labels.mha', '_labels.mha')]
    path = next((p for p in options if p.is_file()), None)
    if path is None:
        raise ValueError('Full validation masks missing')
    full = DS._read_mask_mha(path)
    first = DS._read_mask_mha(case/'targets'/(job['Case']+'_first_label.mha'))
    if len(full) != native_frames or full.shape[1:] != first.shape[1:]:
        raise ValueError('Truth geometry mismatch')
    return F.interpolate(full[:frames, None].float(), (RULE['height'], RULE['width']), mode='nearest')[:, 0].numpy() > .5, path.name


def verify_rule(p, outputs, queries, slot, threshold, mirrored):
    """Check statistics, weights AND targets against the original executed loss hook."""
    import torch
    import cotracker.utils.offline_teacher_policy as policy
    from cotracker.models.core.cotracker.losses import sequence_loss
    tensors = [tuple(torch.from_numpy(x) for x in item) for item in outputs]
    captured, real_sha = [], policy.tensor_sha
    def capture(x):
        captured.append(x.detach().cpu().numpy().copy())
        return real_sha(x)
    primary = tensors[slot]
    for arm in (('gate2', 'fuse2') if threshold == 2 else ('gate4',)):
        coordinator = p.Coordinator(arm, p.matrix()[arm])
        coordinator.context = {'primary': primary, 'reviewers': [x for i, x in enumerate(tensors) if i != slot],
                               'queries': torch.from_numpy(queries), 'height': RULE['height'], 'width': RULE['width'],
                               'slot': slot, 'audit': {}}
        captured.clear()
        with patch.object(policy, 'tensor_sha', side_effect=capture):
            coordinator.loss(sequence_loss, [[primary[0]]], [primary[0]], [torch.ones_like(primary[1])],
                             vis=[(primary[1] > .9).float()], add_huber_loss=True, loss_only_for_visible=True)
        stats = coordinator.context['audit']
        expected = mirrored['fused'] if arm == 'fuse2' else mirrored['primary']
        if (stats['eligible'] != int(mirrored['eligible'].sum()) or
            stats['conflict'] != int(mirrored['conflict'].sum()) or
            not np.array_equal(captured[1], mirrored['weights']) or
            not np.allclose(captured[0], expected, rtol=0, atol=1e-5)):
            raise ValueError('Diagnostic rule differs from actual collaboration: ' + arm)


def rows_for(p, job, outputs, queries, truth, masks):
    import torch
    rows = []
    h, w = RULE['height'], RULE['width']
    for slot in range(3):
        for threshold in RULE['thresholds']:
            r = M.review(outputs, queries, h, w, slot, threshold)
            verify_rule(p, outputs, queries, slot, threshold, r)
            predictions = {name: r[k] for name, k in [('Primary', 'primary'), ('ReviewerA', 'reviewer_a'),
                           ('ReviewerB', 'reviewer_b'), ('ReviewerMean', 'reviewer_mean'), ('Fused', 'fused')]}
            errors = M.known_errors(predictions, truth, h, w)[0] if truth is not None else M.boundary_errors(predictions, masks)
            shuffled = np.zeros_like(r['conflict'])
            gen = torch.Generator().manual_seed(p.local_seed(torch.from_numpy(queries), slot)+1)
            for t in range(r['conflict'].shape[1]):
                ids = np.flatnonzero(r['eligible'][0, t])
                n = int(r['conflict'][0, t].sum())
                shuffled[0, t, ids[torch.randperm(len(ids), generator=gen).numpy()[:n]]] = True
            if not np.array_equal(shuffled.sum(-1), r['conflict'].sum(-1)):
                raise ValueError('Shuffled control count mismatch')
            cohorts = {'eligible': r['eligible'], 'conflict': r['conflict'], 'nonconflict': r['eligible'] & ~r['conflict'],
                       'shuffled': shuffled, 'agreement': r['agreement'], 'fused_changed': r['changed']}
            count = queries.shape[1]
            groups = {'all': np.ones(count, dtype=bool)}
            if truth is not None:
                groups.update(contour=np.arange(count) < count-64, grid=np.arange(count) >= count-64)
            for group, which in groups.items():
                for cohort, selected in cohorts.items():
                    rows.append({'Mode': job['Mode'], 'Case': job['Case'], 'Motion': job['Motion'],
                                 'QueryGroup': group, 'PrimarySlot': slot, 'Threshold': threshold,
                                 'Cohort': cohort, **M.summarize(selected & which[None, None], errors)})
    return rows


def model_hash(models):
    h = hashlib.sha256()
    for model in models:
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def probe(root, datasets, weights, p, scope):
    import torch
    from cotracker.models.build_cotracker import build_cotracker
    if not torch.cuda.is_available():
        raise ValueError('GPU required for teacher probes; no CPU fallback')
    jobs = job_list(read(root/'manifest.json')['case_ids'], scope)
    pending = [j for j in jobs if cache(root, j) is None]
    if not pending:
        print('CACHE VERIFIED ' + scope, flush=True)
    else:
        torch.manual_seed(20261022)
        models = [build_cotracker(str(weights/s['checkpoint']), offline=i == 2, window_len=60 if i == 2 else 16,
                                 v2=i == 0).cuda().eval().requires_grad_(False) for i, s in enumerate(p.B.baseline())]
        initial = model_hash(models)
        for index, job in enumerate(pending):
            started = time.perf_counter()
            data = datasets/('train' if job['Mode'] == 'synthetic' else 'validation')
            video, q, truth, transforms, native_frames = make_input(data, job)
            torch.cuda.reset_peak_memory_stats()
            gpu_video, gpu_q = video.cuda(), q.cuda()
            outputs, evidence = [], []
            for slot, model in enumerate(models):
                draw = p.support_for(slot, gpu_video, gpu_q)
                with torch.inference_mode():
                    result = p.B.teacher_forward(slot, p.TEACHER_TYPES[slot], model, gpu_video, gpu_q,
                                                 draw, 'control', p.B.baseline())
                arrays = tuple(x.detach().cpu().numpy() for x in result[:3])
                if any(not np.isfinite(x).all() for x in arrays):
                    raise ValueError('Nonfinite teacher output')
                outputs.append(arrays)
                evidence.append(result[3])
                del result
            del gpu_video, gpu_q
            peak = torch.cuda.max_memory_reserved() / 2**30
            if peak > 20:
                raise ValueError('Teacher probe exceeded 20 GiB reserved; preserve results')
            masks, mask_name = None, None
            if job['Mode'] == 'real':  # Future GT opened only AFTER all teacher predictions.
                masks, mask_name = real_masks(data, job, video.shape[1], native_frames)
            rows = rows_for(p, job, outputs, q.numpy(), truth, masks)
            if model_hash(models) != initial or any(m.training or any(x.requires_grad for x in m.parameters()) for m in models):
                raise ValueError('Teacher parameters or modes changed')
            folder = root/'jobs'/key(job)
            folder.mkdir(parents=True, exist_ok=True)
            arrays = {f'{s}_{n}': x for s, item in enumerate(outputs) for n, x in zip(('P', 'V', 'C'), item)}
            arrays['queries'] = q.numpy()
            if truth is not None:
                arrays.update(truth=truth, transforms=transforms)
            tmp = folder/'outputs.tmp.npz'
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, folder/'outputs.npz')
            save(folder/'rows.json', rows)
            save(folder/'proof.json', {'job': job, 'rules_match_original': True, 'teacher_evidence': evidence,
                 'frames': video.shape[1], 'native_frames': native_frames, 'real_truth_file': mask_name,
                 'queries': q.shape[1],
                 'peak_reserved_gib': peak, 'seconds': time.perf_counter()-started, 'teacher_parameters_exact': True,
                 'teacher_tensor_sha256': initial})
            save(folder/'verified.json', {'job': job, 'manifest_sha256': sha(root/'manifest.json'),
                 'files': {n: sha(folder/n) for n in ('outputs.npz', 'rows.json', 'proof.json')}})
            print(f'COMMITTED {key(job)} [{index+1}/{len(pending)}] seconds={time.perf_counter()-started:.1f}', flush=True)
            del video, outputs, arrays, masks
        if model_hash(models) != initial or any(m.training or any(x.requires_grad for x in m.parameters()) for m in models):
            raise ValueError('Teacher parameters or modes changed')
        save(root/(scope+'-teacher-freeze.json'), {'complete': True, 'tensor_sha256': initial, 'jobs_run': len(pending)})
    # This gate is a completeness/runtime check, NOT an accuracy/adoption criterion.
    frozen(root/(scope+'-complete.json'), {'complete': True, 'manifest_sha256': sha(root/'manifest.json'),
           'jobs': {key(j): sha(cache(root, j)/'verified.json') for j in jobs}})
    print('PROBES COMPLETE ' + scope, flush=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No diagnostic rows')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def report(root):
    manifest = read(root/'manifest.json')
    gate = read(root/'full-complete.json')
    jobs, rows = job_list(manifest['case_ids'], 'full'), []
    if gate['manifest_sha256'] != sha(root/'manifest.json') or len(gate['jobs']) != len(jobs):
        raise ValueError('Full probes incomplete')
    for job in jobs:
        folder = cache(root, job)
        if folder is None or gate['jobs'][key(job)] != sha(folder/'verified.json'):
            raise ValueError('Full job changed')
        rows.extend(read(folder/'rows.json'))
    write_csv(root/'case-diagnostics.csv', rows)
    summaries = M.aggregate(rows)
    for mode in ('synthetic', 'real'):
        write_csv(root/(mode+'-summary.csv'), [r for r in summaries if r['Mode'] == mode])
    write_csv(root/'mechanism-summary.csv', M.contrasts(rows))
    save(root/'summary.json', {'complete': True, 'jobs': len(jobs), 'train_cases': 40, 'validation_cases': 10,
         'test_cases_read': 0, 'student_training_steps': 0, 'rule': RULE, 'note': NOTE,
         'reports_sha256': {n: sha(root/n) for n in ('case-diagnostics.csv', 'synthetic-summary.csv', 'real-summary.csv', 'mechanism-summary.csv')}})
    print('REPORT COMPLETE ' + str(root/'summary.json'), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['plan', 'smoke', 'full', 'report'])
    for name, default in [('root', '/study'), ('prior', '/prior'), ('collaboration', '/collaboration'),
                          ('datasets', '/datasets'), ('weights', '/checkpoints')]:
        parser.add_argument('--'+name, type=Path, default=Path(default))
    a = parser.parse_args()
    p = prepare(a.root, a.prior, a.collaboration, a.datasets, a.weights)
    if a.command in ('smoke', 'full'):
        if a.command == 'full':
            smoke = read(a.root/'smoke-complete.json')
            if smoke['manifest_sha256'] != sha(a.root/'manifest.json'):
                raise ValueError('Smoke gate changed')
            for job in job_list(read(a.root/'manifest.json')['case_ids'], 'smoke'):
                folder = cache(a.root, job)
                if folder is None or smoke['jobs'].get(key(job)) != sha(folder/'verified.json'):
                    raise ValueError('Missing or changed smoke output')
        probe(a.root, a.datasets, a.weights, p, a.command)
    elif a.command == 'report':
        report(a.root)


if __name__ == '__main__':
    main()
