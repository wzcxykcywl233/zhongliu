"""Isolated patient082 ablation, retaining the original full-study implementation."""
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
from exclusion import POLICY, make_schedule
from source_provenance import verify_historical_sources

ARMS = ('R1', 'B1')
SEEDS = (0, 1, 2)
SEED_INDEX = int(os.environ.get('MCT_FULL_SEED', '0'))
if SEED_INDEX not in SEEDS:
    raise ValueError('Seed must be 0, 1 or 2')
S.STUDENT_SEED = 20261008 + SEED_INDEX
S.SEED = 20261023 + SEED_INDEX
PREFIX = S.PREFIX
PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
RUNTIME = {'backend': 'compact', 'point_chunk_size': 0, 'gain': 1.,
           'max_correction_pixels': 0., 'checkpoint_blocks': False}
RULE = {'version': 1, 'arms': list(ARMS), 'seed': SEED_INDEX, 'student_seed': S.STUDENT_SEED,
        'data_seed': S.SEED, 'external_steps': 3000, 'transfer_steps': 1000, 'smoke_steps': 10,
        'external_lr': {'B1': 1e-6}, 'external_scheduler': 'constant', 'external_clip': 'native min(16,T)',
        'external_queries': 128, 'transfer_frames': 60, 'transfer_queries': 128,
        'transfer_lr': 5e-5, 'transfer_scheduler_num_steps': 1000, 'train_iterations': 4,
        'transfer_trainability': 'Mamba only; own initialized backbone frozen including buffers',
        'confidence_loss': False, 'mask_loss': False, 'auxiliary_teacher': False,
        'profile': PROFILE, 'test_access': 'explicit test stage only, after validation freeze', 'formal_training': True,
        'selection': 'Paired mean DSC>0, D98>=-.002, HD95<=.02; DSC improves in >=2/3 seeds; worst DSC>=-.002',
        'confirmation': {'minimum_DSC_improved_seeds': 2, 'maximum_single_seed_DSC_drop': .002},
        'note': 'Three paired seeds; extra external data/updates, not equal-total-compute attribution.'}
RULE['exclusion'] = POLICY


def baseline_binding(plan, baseline=Path('/baseline')):
    """Only manifests and external training receipts; no evaluation/test data."""
    old = read(baseline / 'manifest.json')
    expected = {k: v for k, v in plan['rule'].items() if k != 'exclusion'}
    if old['rule'] != expected:
        raise ValueError('Historical full study budget/seed differs')
    for key in ('weights_sha256', 'images', 'teacher_pool', 'external_manifest_sha256',
                'trackrad_ids', 'trackrad_files', 'student_config'):
        if old[key] != plan[key]:
            raise ValueError('Historical full study inputs differ: ' + key)
    source_verification = verify_historical_sources(CHECKOUT, old['source_sha256'], S.safe_file)
    receipt = read(baseline / 'full/external/B1/verified.json')
    if (receipt.get('complete') is not True or receipt.get('steps') != RULE['external_steps'] or
            receipt.get('manifest_sha256') != digest(baseline / 'manifest.json') or
            not receipt.get('frozen_tensors_exact')):
        raise ValueError('Completed original 3000-step B1 required')
    paths = ['manifest.json', 'full/external/B1/verified.json']
    for name in ('steps.json', 'initial.json'):
        rel = 'full/external/B1/' + name
        verify_files(baseline, {rel: receipt['files_sha256'][rel]})
        paths.append(rel)
    records = read(baseline / 'full/external/B1/steps.json')
    if [r['step'] for r in records] != list(range(RULE['external_steps'])):
        raise ValueError('Historical external steps incomplete')
    return {**{p: digest(baseline / p) for p in paths}, 'source_verification': source_verification}


def baseline_steps(baseline=Path('/baseline')):
    return read(baseline / 'full/external/B1/steps.json')


def baseline_initial(baseline=Path('/baseline')):
    return read(baseline / 'full/external/B1/initial.json')['state_hashes']


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
    if split not in ('train-40', 'validation-10', 'test-38'):
        raise ValueError('Unknown data split')
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


def checked_plan(root, expected_seed=None):
    seed = SEED_INDEX if expected_seed is None else expected_seed
    if seed not in SEEDS:
        raise ValueError('Invalid frozen seed')
    expected_rule = dict(RULE, seed=seed, student_seed=20261008 + seed, data_seed=20261023 + seed)
    plan = read(root / 'manifest.json')
    if plan['rule'] != expected_rule or plan['teacher_pool'] != T.baseline():
        raise ValueError('Frozen experiment changed')
    verify_files(CHECKOUT, plan['source_sha256'])
    if read(root / 'runtime.json') != RUNTIME:
        raise ValueError('Changed branch runtime')
    if digest(root / 'exclusion-schedule.json') != plan['exclusion_schedule_sha256']:
        raise ValueError('Exclusion schedule changed')
    return plan


def prepare(root, prior_root, audit_root, external_smoke, weights, train, validation, images, pilot):
    pilot_plan = read(pilot / 'manifest.json')
    verify_files(CHECKOUT, pilot_plan['source_sha256'])
    pilot_proof = marker(pilot, 'summary.json')
    if (pilot_proof is None or pilot_proof['stage'] != 'single-seed-pilot' or
            pilot_proof['external_steps'] != 500 or pilot_proof['transfer_steps'] != 250 or
            marker(pilot, 'smoke/verified.json') is None):
        raise ValueError('Completed paired pilot and handoff smoke required')
    candidate = read(pilot / 'reports/validation-10/selection.json')
    if 'B1' not in candidate['eligible_routes']:
        raise ValueError('B1 was not a validation-eligible pilot route')
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
    sources.update(pilot_plan['source_sha256'])
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
    if (pilot_plan['images'] != images or pilot_plan['weights_sha256'] != old['weights_sha256'] or
            pilot_plan['teacher_pool'] != T.baseline() or pilot_plan['trackrad_files'] != inventories or
            pilot_plan['external_manifest_sha256'] != old['data_manifest_sha256']):
        raise ValueError('Full study inputs differ from completed pilot')
    plan = {'schema': 1, 'rule': RULE,
            'pilot_summary_sha256': digest(pilot / 'summary.json'),
            'pilot_selection_sha256': digest(pilot / 'reports/validation-10/selection.json'),
            'prior_inventory_sha256': digest(prior_root / 'frozen-study.json'), 'source_sha256': sources, 'images': images,
            'weights_sha256': old['weights_sha256'], 'teacher_pool': T.baseline(),
            'external_manifest_sha256': old['data_manifest_sha256'],
            'external_smoke_summary_sha256': digest(external_smoke / 'summary.json'),
            'trackrad_ids': ids, 'trackrad_files': inventories, 'student_config': student_config,
            'test_policy': 'No test resolution/mount during all/full/smoke; explicit test only after frozen validation selection'}
    schedule = make_schedule(read(audit_root / 'data-manifest.json'), S.SEED, RULE['external_steps'])
    S.frozen(root / 'exclusion-schedule.json', schedule)
    plan['exclusion_schedule_sha256'] = digest(root / 'exclusion-schedule.json')
    plan['baseline_binding'] = baseline_binding(plan)
    baseline_rows = baseline_steps()
    if any((a['original_patient'], a['slot']) != (b['patient'], b['slot']) for a, b in zip(schedule['rows'], baseline_rows)):
        raise ValueError('Exclusion schedule is not paired with historical full B1')
    S.frozen(root / 'manifest.json', plan)
    S.frozen(root / 'runtime.json', RUNTIME)
    S.frozen(root / 'reference-images.json', {'trackrad-algorithm-cotracker-algorithm': images['Inference'],
                                            'trackrad-evaluation': images['Evaluation']})
    from freeze_sidecar_reference import freeze
    freeze(weights / 'scaled_offline.pth', root / 'pretrained-frozen.pth')
    print(f'PLAN VERIFIED seed={SEED_INDEX}: exclude patient082; 79 train/20 dev; '
          f'{schedule["replacement_steps"]}/3000 replacement steps; R1/B1, no automatic test', flush=True)


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
    if phase not in ('smoke', 'full') or arm not in ARMS:
        raise ValueError('Unsupported arm/stage')
    return {'policy': arm, 'phase': phase, 'steps': RULE['smoke_steps'] if phase == 'smoke' else RULE['transfer_steps'],
            'seed': SEED_INDEX, 'student_seed': S.STUDENT_SEED,
            'pool': T.baseline(), 'pool_sha256': T.digest(T.baseline()),
            'manifest_sha256': digest(root / 'manifest.json'),
            'init_sha256': digest(root / phase / 'init' / (arm + '.pth'))}
