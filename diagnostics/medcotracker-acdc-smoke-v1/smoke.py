"""Bounded real-ACDC GPU smoke, not a pilot, transfer result, or resumable trainer."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import uuid

import torch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parent / 'medcotracker-acdc-ab-v1'))
from acdc_data import AUDIT_VERSION, clip_policy, digest, read, save, split_patients
from cine_dataset import ACDCCineDataset
from ab_modes import PREFIX, ROUTES, configure

SEED = 20261023
STUDENT_SEED = 20261008
TEACHER_TYPES = ('online', 'online_cotracker_three', 'offline_cotracker_three')
LR = {'A1': 5e-5, 'B1': 1e-6, 'B2': 1e-6, 'B3': 5e-7}
RULE = {'version': 1, 'steps_per_route': 5, 'routes': list(ROUTES), 'queries': 128,
        'data_query_seed': SEED, 'student_sidecar_seed': STUDENT_SEED,
        'train_iterations': 4, 'learning_rates': LR, 'optimizer': 'AdamW',
        'weight_decay': 1e-5, 'clip_gradient_norm': 10., 'scheduler': 'none for five-step smoke',
        'loss': 'existing sequence_loss: .05 visible Huber(delta6) + .01 invisible L1; gamma .8',
        'confidence_loss': False, 'visibility_loss': False, 'mask_loss': False,
        'supervision': 'fixed teacher pseudo-tracks, visibility threshold .9; C recorded only',
        'data_padding': False, 'teacher_internal_window_padding': 'unmodified upstream implementation',
        'transfer_training': False, 'pilot_training': False, 'test_access': False,
        'retry': 'verified completed routes reused; interrupted route restarts in a NEW attempt directory'}


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_hashes(model):
    return {k: tensor_sha(v) for k, v in model.state_dict().items()}


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen inputs changed; use NEW OutputRoot: ' + str(path))
    else:
        save(path, value)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def safe_file(root, relative):
    if not isinstance(relative, str) or ':' in relative or '\\' in relative:
        raise ValueError('Unsafe relative filename')
    result = (root / relative).resolve()
    if root.resolve() not in result.parents:
        raise ValueError('Filename escapes root')
    return result


def make_jobs(manifest):
    if (manifest.get('audit_version') != AUDIT_VERSION or manifest.get('training_ready') is not True or
            manifest.get('clip') != clip_policy() or
            manifest.get('split_counts') != {'external-train-80': 80, 'external-dev-20': 20}):
        raise ValueError('Need complete native-v2 80/20 audit')
    rows = manifest['patients']
    expected = split_patients(rows)
    if any(r['split'] != expected[r['patient']] for r in rows):
        raise ValueError('Patient split changed')
    train = [r for r in rows if r['split'] == 'external-train-80']
    shortest = min(train, key=lambda r: (r['shape_tzyx'][0], r['patient']))
    if shortest['shape_tzyx'][0] != min(r['shape_tzyx'][0] for r in rows):
        raise ValueError('Shortest cine is dev-only; design a separate non-training interface probe first')
    long = min((r for r in train if r['shape_tzyx'][0] >= 16), key=lambda r: r['patient'])
    # Three interfaces at the shortest native length, plus three at 16 frames.
    # First five are the SAME training inputs/labels for every route; sixth is interface-only.
    choices = [(shortest, 0), (shortest, 1), (shortest, 2), (long, 0), (long, 2), (long, 1)]
    return [{'index': i, 'patient': row['patient'], 'slot': slot,
             'frames': min(16, row['shape_tzyx'][0]), 'train_step': i < 5,
             'query_seed': SEED + 101 * i} for i, (row, slot) in enumerate(choices)]


def load_protocol(checkout):
    path = checkout / 'diagnostics/teacher-combination-search-v4/protocol.py'
    spec = importlib.util.spec_from_file_location('acdc_fixed_teachers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(checkout, manifest_path, weights, root, protocol):
    manifest = read(manifest_path)
    jobs = make_jobs(manifest)
    sources = {}
    for folder in (checkout / 'cotracker-algorithm/ext/co-tracker/cotracker', PACKAGE,
                   PACKAGE.parent / 'medcotracker-acdc-ab-v1'):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.ps1', '.txt', '.json'):
                sources[path.relative_to(checkout).as_posix()] = digest(path)
    for relative in ('diagnostics/teacher-combination-search-v4/protocol.py',):
        sources[relative] = digest(checkout / relative)
    image = os.environ.get('ACDC_SMOKE_IMAGE_ID', '')
    if len(image) != 71 or not image.startswith('sha256:'):
        raise ValueError('Missing pinned runtime image')
    value = {'schema': 1, 'rule': RULE, 'data_manifest_sha256': digest(manifest_path),
             'weights_sha256': {s['checkpoint']: digest(weights / s['checkpoint']) for s in protocol.baseline()},
             'source_sha256': sources, 'image': image, 'jobs': jobs, 'teacher_pool': protocol.baseline(),
             'torch': str(torch.__version__), 'cuda': torch.version.cuda,
             'device': torch.cuda.get_device_name(),
             'scope': 'ACDC external-train only; dev/test/TrackRAD not mounted as training inputs'}
    frozen(root / 'frozen-smoke.json', value)
    return jobs


def get_sample(dataset, job):
    index = next(i for i, r in enumerate(dataset.rows) if r['patient'] == job['patient'])
    sample = dataset[index]
    if sample['video'].shape[0] != job['frames']:
        raise ValueError('Native clip length changed')
    return sample


def load_verified(root, marker_name):
    marker = root / marker_name
    if not marker.exists():
        return None
    report = read(marker)
    if not report.get('complete') or report['plan_sha256'] != digest(root / 'frozen-smoke.json'):
        raise ValueError('Stale/incomplete cache marker')
    for relative, expected in report['files_sha256'].items():
        if digest(safe_file(root, relative)) != expected:
            raise ValueError('Cached artifact changed: ' + relative)
    return report


def prepare_labels(dataset, jobs, weights, root, protocol):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_points_on_a_grid, get_uniformly_sampled_pts
    cached = load_verified(root, 'labels-verified.json')
    if cached:
        print('CACHE VERIFIED teacher labels (all six interfaces)', flush=True)
        return cached
    attempt = root / 'attempts' / ('labels-' + uuid.uuid4().hex[:10])
    records, files = [], {}
    for job in jobs:
        sample = get_sample(dataset, job)
        video = sample['video'][None].cuda()
        torch.manual_seed(job['query_seed'])
        q = get_uniformly_sampled_pts(128, video.shape[1], video.shape[-2:], device='cpu').cuda()
        if job['slot'] < 2:
            xy = get_points_on_a_grid(5, video.shape[-2:], device='cpu')
            draw = torch.cat((torch.zeros_like(xy[..., :1]), xy), -1).cuda()
        else:
            draw = get_uniformly_sampled_pts(100, video.shape[1], video.shape[-2:], device='cpu').cuda()
        slot = job['slot']
        model = build_cotracker(str(weights / protocol.baseline()[slot]['checkpoint']),
                               offline=slot == 2, v2=slot == 0,
                               window_len=60 if slot == 2 else 16).cuda().eval().requires_grad_(False)
        before = state_hashes(model)
        p, v, c, audit = protocol.teacher_forward(slot, TEACHER_TYPES[slot], model, video, q,
                                                 draw, 'control', protocol.baseline())
        if state_hashes(model) != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Teacher weights/gradients changed')
        if not bool((v > .9).any()):
            raise ValueError('No visible pseudo-labels; inspect, do not silently skip')
        path = attempt / (str(job['index']) + '.pth')
        value = {'job': job, 'sampling': sample['sampling'], 'video_sha256': tensor_sha(video),
                 'queries': q.cpu(), 'tracks': p.cpu(), 'visibility': v.cpu(), 'confidence': c.cpu(),
                 'audit': audit, 'teacher_tensors_exact': True}
        atomic_torch(path, value)
        relative = path.relative_to(root).as_posix()
        files[relative] = digest(path)
        records.append({'job': job, 'file': relative, 'video_sha256': value['video_sha256'],
                        'queries_sha256': tensor_sha(q), 'teacher_seconds': audit['teacher_seconds']})
        print(f"TEACHER INTERFACE PASSED {TEACHER_TYPES[slot]} {job['patient']} T={video.shape[1]}", flush=True)
        del model, video, q, draw, p, v, c
        torch.cuda.empty_cache()
    report = {'complete': True, 'plan_sha256': digest(root / 'frozen-smoke.json'),
              'records': records, 'files_sha256': files}
    save(root / 'labels-verified.json', report)
    return report


def training_loss(output, targets):
    from cotracker.models.core.cotracker.losses import sequence_loss
    predictions, _, _, valid = output[3]
    tracks, vis = targets['tracks'], (targets['visibility'] > .9).float()
    if len(predictions) != 1 or valid.shape != vis.shape:
        raise ValueError('Expected one native offline clip')
    common = dict(flow_preds=predictions, flow_gt=[tracks], valids=[valid],
                  gamma=.8, loss_only_for_visible=True)
    return (.05 * sequence_loss(**common, vis=[vis], add_huber_loss=True) +
            .01 * sequence_loss(**common, vis=[1 - vis], add_huber_loss=False))


def new_model(weights, route):
    from cotracker.models.build_cotracker import build_cotracker
    torch.manual_seed(STUDENT_SEED)
    model = build_cotracker(str(weights / 'scaled_offline.pth'), offline=True, window_len=60)
    # Construction consumes RNG; reset before every new sidecar, including the handoff control.
    torch.manual_seed(STUDENT_SEED)
    report = configure(model, route)
    return model, report


def reload_checkpoint(path, route):
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['route'] != route or checkpoint['steps'] != 5:
        raise ValueError('Wrong external checkpoint')
    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
    configure(model, route)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


def configure_handoff(model, route):
    from cotracker.models.core.cotracker.feature_sidecar import (
        attach_sidecar, configure_sidecar_runtime, set_sidecar_training_mode)
    before = state_hashes(model)
    if route != 'A1':
        torch.manual_seed(STUDENT_SEED)
        attach_sidecar(model, 'mamba')
        model.updateformer.feature_sidecar.to(next(model.fnet.parameters()).device)
    model.frozen_pretrained = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIX))
    configure_sidecar_runtime(model, backend='compact')
    set_sidecar_training_mode(model)
    after = state_hashes(model)
    if any(after[n] != h for n, h in before.items()):
        raise ValueError('Handoff changed existing tensors')
    return {'backbone_preserved_exact': True, 'sidecar_carried': route == 'A1',
            'sidecar_trainable_only': True,
            'sidecar_hashes': {n: h for n, h in after.items() if n.startswith(PREFIX)}}


def check_changes(model, before, trainable):
    after = state_hashes(model)
    if set(after) != set(before):
        raise ValueError('State keys changed')
    changed = [n for n in before if before[n] != after[n]]
    unexpected = set(changed) - set(trainable)
    if unexpected or not changed:
        raise ValueError('Frozen tensors changed or no update: ' + str(sorted(unexpected)))
    return changed


def run_route(dataset, labels, weights, root, route):
    marker = route + '-verified.json'
    cached = load_verified(root, marker)
    if cached:
        print('CACHE VERIFIED ' + route, flush=True)
        return cached
    attempt = root / 'attempts' / (route + '-' + uuid.uuid4().hex[:10])
    attempt.mkdir(parents=True)
    model, configuration = new_model(weights, route)
    initial = state_hashes(model)
    model.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LR[route], weight_decay=RULE['weight_decay'], eps=1e-8)
    records = []
    for record in labels['records'][:5]:
        label = torch.load(safe_file(root, record['file']), map_location='cpu', weights_only=True)
        sample = get_sample(dataset, label['job'])
        video = sample['video'][None]
        if tensor_sha(video) != label['video_sha256'] or sample['sampling'] != label['sampling']:
            raise ValueError('Paired clip/sampling changed')
        q = label['queries'].cuda()
        target = {k: label[k].cuda() for k in ('tracks', 'visibility')}
        video = video.cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(video=video, queries=q, iters=4, is_train=True)
        loss = training_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite pseudo-label loss')
        loss.backward()
        gradients = [n for n, p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)]
        if not gradients or set(gradients) - set(configuration['trainable_names']):
            raise ValueError('Wrong/no gradient path')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Nonfinite gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        records.append({'step': len(records), 'input': record, 'loss': float(loss.detach()),
                        'gradient_norm': float(norm), 'gradient_tensors': len(gradients),
                        'seconds': seconds, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})
        save(attempt / 'partial-steps.json', records)
        print(f"SMOKE {route} step={len(records)}/5 {sample['patient']} T={video.shape[1]} seconds={seconds:.1f}", flush=True)
        if seconds > 120 or torch.cuda.max_memory_reserved() > 22 * 2**30:
            raise RuntimeError('Smoke time/memory guard exceeded; preserve attempt, no larger training')
        del prediction, loss, target
    changed = check_changes(model, initial, configuration['trainable_names'])
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite trained parameters')
    final = attempt / 'external-smoke.pth'
    atomic_torch(final, {'schema': 1, 'route': route, 'steps': 5, 'model': model.cpu().state_dict(),
                         'optimizer': optimizer.state_dict(), 'plan_sha256': digest(root / 'frozen-smoke.json'),
                         'feature_sidecar': 'mamba' if route == 'A1' else 'none', 'backbone_architecture': 'base'})
    del optimizer
    model.cuda().eval()
    with torch.no_grad():
        original = tuple(x.cpu() for x in model(video=video, queries=q, iters=4)[:3])
    final_hashes = state_hashes(model)
    del model
    torch.cuda.empty_cache()
    loaded = reload_checkpoint(final, route)
    if state_hashes(loaded) != final_hashes:
        raise ValueError('Strict checkpoint reload changed tensors')
    loaded.cuda().eval()
    with torch.no_grad():
        reloaded = tuple(x.cpu() for x in loaded(video=video, queries=q, iters=4)[:3])
    differences = [float((a - b).abs().max()) for a, b in zip(original, reloaded)]
    if any(not bool(torch.isfinite(b).all()) or not torch.allclose(a, b, rtol=1e-5, atol=1e-4)
           for a, b in zip(original, reloaded)):
        raise ValueError('Reload inference changed: ' + str(differences))
    handoff = configure_handoff(loaded, route)
    # This is a graph/handoff check on ACDC, NOT a TrackRAD training step.
    loaded.zero_grad(set_to_none=True)
    p, v, c = loaded(video=video, queries=q, iters=2, is_train=True)[:3]
    (p.square().mean() * .001 + v.square().mean() + c.square().mean()).backward()
    handoff_gradients = [n for n, p in loaded.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)]
    if not handoff_gradients or any(not n.startswith(PREFIX) for n in handoff_gradients):
        raise ValueError('Handoff sidecar gradient missing or leaking into backbone')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in loaded.parameters()):
        raise ValueError('Handoff nonfinite gradient')
    handoff.update(gradient_path_verified=True, optimizer_steps=0, data='same ACDC clip, not TrackRAD')
    report = {'complete': True, 'plan_sha256': digest(root / 'frozen-smoke.json'), 'route': route,
              'configuration': configuration, 'records': records, 'changed_tensors': changed,
              'initial_sidecar_hashes': {n: h for n, h in initial.items() if n.startswith(PREFIX)},
              'frozen_tensors_exact': True, 'strict_reload_tensors_exact': True,
              'reload_inference_max_abs_differences': differences, 'handoff': handoff,
              'files_sha256': {p.relative_to(root).as_posix(): digest(p)
                               for p in (final, attempt / 'partial-steps.json')}}
    save(root / marker, report)
    del loaded, video, q, p, v, c
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, default=Path('/checkout'))
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--audit', type=Path, default=Path('/audit'))
    parser.add_argument('--weights', type=Path, default=Path('/weights'))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA smoke required; no CPU fallback')
    if os.environ.get('TRACKRAD_SIDECAR_RUNTIME') or os.environ.get('COTRACKER_CHECKPOINT'):
        raise ValueError('Unexpected model/runtime override')
    sys.path.insert(0, str(args.checkout / 'cotracker-algorithm/ext/co-tracker'))
    import cotracker
    if args.checkout.resolve() not in Path(cotracker.__file__).resolve().parents:
        raise ValueError('Wrong installed CoTracker source; use read-only checkout')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = load_protocol(args.checkout)
    manifest_path = args.audit / 'data-manifest.json'
    jobs = prepare(args.checkout, manifest_path, args.weights, args.output, protocol)
    dataset = ACDCCineDataset(args.data, manifest_path)
    print(f"GPU SMOKE: min-native={jobs[0]['frames']}; {jobs[0]['patient']}; 4 routes x 5 steps", flush=True)
    labels = prepare_labels(dataset, jobs, args.weights, args.output, protocol)
    reports = [run_route(dataset, labels, args.weights, args.output, r) for r in ROUTES]
    inputs = [r['input'] for r in reports[0]['records']]
    if any([r['input'] for r in report['records']] != inputs for report in reports):
        raise ValueError('A/B paired labels changed')
    fresh = [r['handoff']['sidecar_hashes'] for r in reports if r['route'] != 'A1']
    if any(branch != fresh[0] for branch in fresh):
        raise ValueError('B route fresh sidecar initializations differ')
    if fresh[0] != reports[0]['initial_sidecar_hashes']:
        raise ValueError('Fresh B sidecar differs from the seed-matched initial A1 sidecar')
    # Verify all published artifacts again before declaring completion.
    for route in ROUTES:
        load_verified(args.output, route + '-verified.json')
    prepare(args.checkout, manifest_path, args.weights, args.output, protocol)
    save(args.output / 'summary.json', {'complete': True, 'stage': 'external-gpu-smoke',
         'routes': list(ROUTES), 'steps_per_route': 5, 'teacher_interfaces': 6,
         'minimum_native_frames': jobs[0]['frames'], 'minimum_native_patient': jobs[0]['patient'],
         'paired_inputs_and_labels': True, 'B_fresh_sidecars_exact': True,
         'frozen_tensors_and_handoff_passed': True, 'trackrad_training_started': False,
         'pilot_started': False, 'route_proofs': {r: digest(args.output / (r + '-verified.json')) for r in ROUTES},
         'note': 'Five-step interface/gradient test, not medical tracking accuracy. Resume is route-level restart, not optimizer continuation.'})
    print('EXTERNAL GPU SMOKE COMPLETE. No TrackRAD training, pilot or test queue started.', flush=True)


if __name__ == '__main__':
    main()
