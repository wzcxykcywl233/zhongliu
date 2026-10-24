"""Bounded, seed-paired ACDC -> TrackRAD pilot. No test-set stage."""
import hashlib
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import random
import sys

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
CHECKOUT = PACKAGE.parents[1]
APP = CHECKOUT / 'cotracker-algorithm'
sys.path[:0] = [str(APP / 'ext/co-tracker'), str(APP), str(APP / 'experiments'),
               str(PACKAGE.parent / 'medcotracker-acdc-smoke-v1')]
import smoke as S
from acdc_data import digest, read, save

ARMS = ('R1', 'A1', 'B1', 'B2', 'B3')
PREFIX = S.PREFIX
PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
RUNTIME = {'backend': 'compact', 'point_chunk_size': 0, 'gain': 1.,
           'max_correction_pixels': 0., 'checkpoint_blocks': False}
RULE = {'version': 1, 'arms': list(ARMS), 'seed': 0, 'student_seed': S.STUDENT_SEED,
        'data_seed': S.SEED, 'external_steps': 500, 'transfer_steps': 250, 'smoke_steps': 10,
        'external_lr': S.LR, 'external_scheduler': 'constant', 'external_clip': 'native min(16,T)',
        'external_queries': 128, 'transfer_frames': 60, 'transfer_queries': 128,
        'transfer_lr': 5e-5, 'transfer_scheduler_num_steps': 1000, 'train_iterations': 4,
        'transfer_trainability': 'Mamba only; own initialized backbone frozen including buffers',
        'confidence_loss': False, 'mask_loss': False, 'auxiliary_teacher': False,
        'profile': PROFILE, 'test_access': False, 'formal_training': False,
        'selection': 'DSC > R1; D98 >= R1-.002; HD95 <= R1+.02; rank DSC,HD95,CD,D98,name',
        'note': 'Extra external data/updates, not equal-total-compute attribution; one seed is not stability evidence.'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


T = S.load_protocol(CHECKOUT)
F = module('transfer_frozen_protocol', PACKAGE.parent / 'frozen-sidecar-teachers-v1/protocol.py')


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One visible CUDA GPU required; no CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def hashes(state):
    return {n: S.tensor_sha(v) for n, v in state.items()}


def verify_files(root, files):
    for relative, expected in files.items():
        if digest(S.safe_file(root, relative)) != expected:
            raise ValueError('Artifact changed: ' + relative)


def marker(root, relative):
    path = root / relative
    if not path.exists():
        return None
    value = read(path)
    if value.get('complete') is not True or value.get('manifest_sha256') != digest(root / 'manifest.json'):
        raise ValueError('Incomplete/stale proof: ' + relative)
    verify_files(root, value['files_sha256'])
    return value


def publish(root, relative, files, **extra):
    value = {'complete': True, 'manifest_sha256': digest(root / 'manifest.json'),
             'files_sha256': {p.relative_to(root).as_posix(): digest(p) for p in files}, **extra}
    save(root / relative, value)
    return value


def validate_inventory(prior):
    counts = {'train-40': 40, 'validation-10': 10, 'test-38': 38}
    ids = {split: sorted({r['Case'] for r in prior['Dataset'] if r['Split'] == split}) for split in counts}
    if any(len(ids[s]) != n for s, n in counts.items()):
        raise ValueError('Wrong 40/10/38 inventory')
    if sum(map(len, ids.values())) != len(set(sum(ids.values(), []))):
        raise ValueError('Overlapping case IDs')
    return ids


def input_files(prior, split):
    if split not in ('train-40', 'validation-10'):
        raise ValueError('No test data access')
    files = {}
    for row in prior['Dataset']:
        if row['Split'] != split:
            continue
        parts = PureWindowsPath(row['File']).parts
        if parts.count(row['Case']) != 1:
            raise ValueError('Ambiguous case path')
        relative = '/'.join(parts[parts.index(row['Case']):])
        if relative in files:
            raise ValueError('Duplicate input file')
        files[relative] = row['SHA256'].lower()
    return files


def verify_dataset(root, files, cases):
    if sorted(p.name for p in root.iterdir() if p.is_dir()) != sorted(cases):
        raise ValueError('Dataset case inventory changed')
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} != set(files):
        raise ValueError('Dataset file inventory changed')
    verify_files(root, files)


def checked_plan(root):
    plan = read(root / 'manifest.json')
    if plan['rule'] != RULE or plan['teacher_pool'] != T.baseline():
        raise ValueError('Frozen experiment changed')
    verify_files(CHECKOUT, plan['source_sha256'])
    if read(root / 'runtime.json') != RUNTIME:
        raise ValueError('Changed branch runtime')
    return plan


def prepare(root, prior_root, audit_root, external_smoke, weights, train, validation, images):
    prior = read(prior_root / 'frozen-study.json')
    ids = validate_inventory(prior)
    ext = read(external_smoke / 'summary.json')
    if (not ext.get('complete') or ext.get('stage') != 'external-gpu-smoke' or
            ext.get('steps_per_route') != 5 or ext.get('teacher_interfaces') != 6 or ext.get('routes') != list(S.ROUTES) or
            not all(ext.get(k) for k in ('paired_inputs_and_labels', 'B_fresh_sidecars_exact',
                                        'frozen_tensors_and_handoff_passed'))):
        raise ValueError('Complete external GPU smoke required')
    old = read(external_smoke / 'frozen-smoke.json')
    verify_files(CHECKOUT, old['source_sha256'])
    verify_files(weights, old['weights_sha256'])
    if digest(audit_root / 'data-manifest.json') != old['data_manifest_sha256']:
        raise ValueError('External audited data changed')
    S.make_jobs(read(audit_root / 'data-manifest.json'))
    if S.load_verified(external_smoke, 'labels-verified.json') is None:
        raise ValueError('Missing external teacher interface proof')
    if old['teacher_pool'] != T.baseline() or images['Training'] != old['image']:
        raise ValueError('Teacher pool/training image differs from successful external smoke')
    for arm in S.ROUTES:
        if ext['route_proofs'][arm] != digest(external_smoke / (arm + '-verified.json')):
            raise ValueError('External smoke proof changed')
        S.load_verified(external_smoke, arm + '-verified.json')
    sources = {r['File'].replace('\\', '/'): r['SHA256'].lower() for r in prior['Source']}
    verify_files(CHECKOUT, sources)
    sources.update(old['source_sha256'])
    # Explicit dependencies not necessarily present in the historical source inventory.
    for folder in (PACKAGE, PACKAGE.parent / 'frozen-sidecar-teachers-v1'):
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.py', '.ps1', '.txt'):
                sources[path.relative_to(CHECKOUT).as_posix()] = digest(path)
    for name in ('summarize_feature_sidecar.py', 'summarize_backbone_growth.py', 'freeze_sidecar_reference.py'):
        relative = 'cotracker-algorithm/experiments/' + name
        sources[relative] = digest(CHECKOUT / relative)
    sources['scripts/run_hierarchical_followup_resumable.ps1'] = digest(CHECKOUT / 'scripts/run_hierarchical_followup_resumable.ps1')
    sources['scripts/experiment_local_runtimes.ps1'] = digest(CHECKOUT / 'scripts/experiment_local_runtimes.ps1')
    verify_files(CHECKOUT, sources)
    inventories = {s: input_files(prior, s) for s in ('train-40', 'validation-10')}
    verify_dataset(train, inventories['train-40'], ids['train-40'])
    verify_dataset(validation, inventories['validation-10'], ids['validation-10'])
    from experiments import get_experiment_config
    student_config = asdict(get_experiment_config(PROFILE)[1])
    if student_config['border_points'] != 1000 or student_config['n_iterations'] != 2:
        raise ValueError('Frozen inference profile changed')
    plan = {'schema': 1, 'rule': RULE, 'source_sha256': sources, 'images': images,
            'weights_sha256': old['weights_sha256'], 'teacher_pool': T.baseline(),
            'external_manifest_sha256': old['data_manifest_sha256'],
            'external_smoke_summary_sha256': digest(external_smoke / 'summary.json'),
            'trackrad_ids': ids, 'trackrad_files': inventories, 'student_config': student_config,
            'test_policy': 'IDs only for disjointness; no test directory resolution/mount/read'}
    S.frozen(root / 'manifest.json', plan)
    S.frozen(root / 'runtime.json', RUNTIME)
    S.frozen(root / 'reference-images.json', {'trackrad-algorithm-cotracker-algorithm': images['Inference'],
                                            'trackrad-evaluation': images['Evaluation']})
    from freeze_sidecar_reference import freeze
    freeze(weights / 'scaled_offline.pth', root / 'pretrained-frozen.pth')
    print('PLAN VERIFIED: R1/A1/B1/B2/B3; external500 + transfer250; seed0; no test access', flush=True)


def instrument(source):
    source = T.instrument(source, F.instrument)
    old = """            architecture_report = warm_start(model, self.load(args.restore_ckpt), args.backbone_architecture)
            if args.feature_sidecar != 'none':
                from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar_verified
                architecture_report['sidecar'] = attach_sidecar_verified(model, args.feature_sidecar)"""
    if source.count(old) != 1:
        raise ValueError('Trainer handoff anchor changed')
    return source.replace(old, '            architecture_report = __initialize__(model, args)')


def config(root, phase, arm):
    if phase not in ('smoke', 'pilot') or arm not in ARMS:
        raise ValueError('Unsupported arm/stage')
    return {'policy': arm, 'phase': phase, 'steps': 10 if phase == 'smoke' else 250,
            'pool': T.baseline(), 'pool_sha256': T.digest(T.baseline()),
            'manifest_sha256': digest(root / 'manifest.json'),
            'init_sha256': digest(root / phase / 'init' / (arm + '.pth'))}


def selection(rows):
    reference = next(r for r in rows if r['Arm'] == 'R1')
    eligible = [r for r in rows if r['Arm'] in S.ROUTES and r['DSC'] > reference['DSC'] and
                r['D98'] >= reference['D98'] - .002 and r['HD95'] <= reference['HD95'] + .02]
    eligible.sort(key=lambda r: (-r['DSC'], r['HD95'], r['CD'], -r['D98'], r['Arm']))
    return [r['Arm'] for r in eligible]
