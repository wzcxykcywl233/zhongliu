"""Deterministic native-cine pretraining, with atomic optimizer continuation."""
import argparse
from pathlib import Path
import time
import shutil

from common import *
from exclusion import EXCLUDED, dataset_class

TEACHERS = {}
HISTORICAL_ROWS = None
BASELINE_LABELS = Path('/baseline/full/labels')


def save_resume(folder, value):
    # Commit the new file BEFORE rotating the previous checkpoint. At every interruption
    # point at least one complete state remains; the uncommitted .next file is ignored.
    new = folder / 'resume.next.pth'
    atomic_torch(new, value)
    current = folder / 'resume.pth'
    if current.exists():
        os.replace(current, folder / 'resume.previous.pth')
    os.replace(new, current)


def load_resume(folder, manifest_sha256, route):
    candidates = [folder / n for n in ('resume.pth', 'resume.previous.pth') if (folder / n).exists()]
    for path in candidates:
        try:
            saved = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as error:
            print('RESUME unreadable ' + str(path) + ': ' + str(error), flush=True)
            continue
        if saved['manifest_sha256'] != manifest_sha256 or saved['route'] != route:
            raise ValueError('Wrong external continuation')
        rows = saved['rows']
        if (saved['steps'] != len(rows) or not 0 <= len(rows) <= RULE['external_steps'] or
                [r['step'] for r in rows] != list(range(len(rows)))):
            raise ValueError('Invalid continuation step history')
        print(f'EXTERNAL RESUME {route} step={len(rows)} from {path.name}', flush=True)
        return saved
    if candidates:
        raise ValueError('No readable external checkpoint; preserve results for diagnosis')
    return None


def schedule(step, length=80):
    epoch, offset = divmod(step, length)
    index = int(np.random.default_rng(S.SEED + epoch).permutation(length)[offset])
    slot = random.Random(S.SEED + 7919 * step).randrange(3)
    return epoch, index, slot


def label_for(root, step, sample, weights):
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.core.model_utils import get_points_on_a_grid, get_uniformly_sampled_pts
    path = root / 'full' / 'labels' / f'{step:04}.pth'
    proof = f'full/labels/{step:04}.json'
    signature = {'step': step, 'patient': sample['patient'], 'sampling': sample['sampling'],
                 'video_sha256': S.tensor_sha(sample['video'][None]), 'slot': schedule(step)[2]}
    planned = read(root / 'exclusion-schedule.json')['rows'][step]
    if (sample['patient'], signature['slot']) != (planned['patient'], planned['slot']) or sample['patient'] == EXCLUDED:
        raise ValueError('Sample differs from exclusion schedule')
    cached = marker(root, proof)
    if cached:
        label = torch.load(path, map_location='cpu', weights_only=True)
        if label['signature'] != signature:
            raise ValueError('Resumed native clip changed')
        return label, digest(path)
    if not planned['replaced']:
        # Preserve exact supervision on every unaffected update. Never use old082 labels.
        original = BASELINE_LABELS / f'{step:04}.pth'
        if HISTORICAL_ROWS is None or digest(original) != HISTORICAL_ROWS[step]['label_sha256']:
            raise ValueError('Historical cached label missing/changed')
        label = torch.load(original, map_location='cpu', weights_only=True)
        if label['signature'] != signature:
            raise ValueError('Unaffected clip changed relative to original B1')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.copy.tmp')
        shutil.copyfile(original, temporary)
        if digest(temporary) != HISTORICAL_ROWS[step]['label_sha256']:
            raise ValueError('Historical label copy changed')
        os.replace(temporary, path)
        publish(root, proof, [path], signature=signature, historical_label_byte_identical=True)
        return label, digest(path)
    reset_rng(S.SEED + 101 * step)
    video = sample['video'][None].cuda()
    q = get_uniformly_sampled_pts(128, video.shape[1], video.shape[-2:], device='cpu').cuda()
    slot = signature['slot']
    if slot < 2:
        xy = get_points_on_a_grid(5, video.shape[-2:], device='cpu')
        draw = torch.cat((torch.zeros_like(xy[..., :1]), xy), -1).cuda()
    else:
        draw = get_uniformly_sampled_pts(100, video.shape[1], video.shape[-2:], device='cpu').cuda()
    if slot not in TEACHERS:
        teacher = build_cotracker(str(weights / T.baseline()[slot]['checkpoint']), offline=slot == 2,
                                 v2=slot == 0, window_len=60 if slot == 2 else 16).cuda().eval().requires_grad_(False)
        TEACHERS[slot] = teacher, S.state_hashes(teacher)
    teacher, before = TEACHERS[slot]
    p, v, c, audit = T.teacher_forward(slot, S.TEACHER_TYPES[slot], teacher, video, q, draw, 'fixed', T.baseline())
    if S.state_hashes(teacher) != before or not bool((v > .9).any()):
        raise ValueError('Teacher changed or no usable visible pseudo-labels')
    label = {'signature': signature, 'queries': q.cpu(), 'tracks': p.cpu(), 'visibility': v.cpu(),
             'confidence': c.cpu(), 'audit': audit}
    atomic_torch(path, label)
    publish(root, proof, [path], signature=signature)
    del teacher, video, p, v, c, q, draw
    torch.cuda.empty_cache()
    return label, digest(path)


def exclusion_smoke(root, data, audit, weights):
    global HISTORICAL_ROWS
    gpu()
    plan = checked_plan(root)
    if baseline_binding(plan) != plan['baseline_binding']:
        raise ValueError('Historical B1 provenance changed')
    verify_files(weights, plan['weights_sha256'])
    if digest(audit / 'data-manifest.json') != plan['external_manifest_sha256']:
        raise ValueError('Audited data changed')
    if marker(root, 'smoke/exclusion-verified.json'):
        print('CACHE VERIFIED exclusion teacher smoke', flush=True)
        return
    HISTORICAL_ROWS = baseline_steps()
    rows = read(root / 'exclusion-schedule.json')['rows']
    chosen = [next(r for r in rows if not r['replaced'])]
    chosen += [next(r for r in rows if r['replaced'] and r['slot'] == slot) for slot in range(3)]
    dataset = dataset_class(S.ACDCCineDataset)(data, audit / 'data-manifest.json', seed=S.SEED)
    files = [root / 'exclusion-schedule.json']
    for row in chosen:
        dataset.set_epoch(row['epoch'])
        sample = dataset[row['original_index']]
        label, _ = label_for(root, row['step'], sample, weights)
        if any(not bool(torch.isfinite(label[k]).all()) for k in ('queries', 'tracks', 'visibility', 'confidence')):
            raise ValueError('Nonfinite exclusion smoke labels')
        files += [root / 'full/labels' / f'{row["step"]:04}{suffix}' for suffix in ('.pth', '.json')]
        print(f'SMOKE exclusion step={row["step"]} patient={sample["patient"]} slot={row["slot"]} replaced={row["replaced"]}', flush=True)
    publish(root, 'smoke/exclusion-verified.json', files, actual_replacements=3,
            unchanged_cached_label=1, teacher_slots=[0, 1, 2], no_student_update=True)


def train(root, data, audit, weights, route):
    global HISTORICAL_ROWS
    if route != 'B1':
        raise ValueError('Only B1 is authorized for full external training')
    gpu()
    plan = checked_plan(root)
    if baseline_binding(plan) != plan['baseline_binding']:
        raise ValueError('Historical B1 provenance changed')
    HISTORICAL_ROWS = baseline_steps()
    if marker(root, 'smoke/verified.json') is None:
        raise ValueError('TrackRAD handoff smoke must pass before external full')
    if marker(root, 'smoke/exclusion-verified.json') is None:
        raise ValueError('Actual replacement teacher smoke must pass before external full')
    verify_files(weights, plan['weights_sha256'])
    if digest(audit / 'data-manifest.json') != plan['external_manifest_sha256']:
        raise ValueError('External manifest changed')
    done = f'full/external/{route}/verified.json'
    if marker(root, done):
        print('CACHE VERIFIED external ' + route, flush=True)
        return
    dataset = dataset_class(S.ACDCCineDataset)(data, audit / 'data-manifest.json', seed=S.SEED)
    model, configuration = S.new_model(weights, route)
    before = S.state_hashes(model)
    if before != baseline_initial():
        raise ValueError('Pretraining must restart from the identical original initialization')
    model.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=S.LR[route], weight_decay=1e-5, eps=1e-8)
    folder = root / 'full/external' / route
    folder.mkdir(parents=True, exist_ok=True)
    S.frozen(folder / 'initial.json', {'state_hashes': before, 'configuration': configuration,
                                     'manifest_sha256': digest(root / 'manifest.json')})
    rows = []
    saved = load_resume(folder, digest(root / 'manifest.json'), route)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        rows = saved['rows']
        for row in rows:
            if marker(root, f"full/labels/{row['step']:04}.json") is None:
                raise ValueError('Prior label proof missing')
            if digest(root / 'full/labels' / f"{row['step']:04}.pth") != row['label_sha256']:
                raise ValueError('Prior labels changed')
    for step in range(len(rows), RULE['external_steps']):
        started = time.perf_counter()
        epoch, index, slot = schedule(step)
        dataset.set_epoch(epoch)
        sample = dataset[index]
        label, label_hash = label_for(root, step, sample, weights)
        # Labels cached before resetting step RNG: uninterrupted/resumed routes have identical RNG.
        reset_rng(S.STUDENT_SEED + step)
        video, q = sample['video'][None].cuda(), label['queries'].cuda()
        targets = {k: label[k].cuda() for k in ('tracks', 'visibility')}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result = model(video=video, queries=q, iters=4, is_train=True)
        loss = S.training_loss(result, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite external loss')
        loss.backward()
        gradients = [n for n, p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().sum() > 0)]
        if not gradients or set(gradients) - set(configuration['trainable_names']):
            raise ValueError('Missing/wrong gradient path')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - update_started
        reserved = torch.cuda.max_memory_reserved() / 2**20
        rows.append({'step': step, 'patient': sample['patient'], 'slot': slot, 'frames': video.shape[1],
                     'label_sha256': label_hash, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                     'update_seconds': seconds, 'wall_seconds_before_checkpoint': time.perf_counter() - started,
                     'peak_reserved_mib': reserved})
        if seconds > 120 or reserved > 22528:
            save(folder / 'failed-step.json', rows[-1])
            raise RuntimeError('External time/memory guard; no larger queue starts')
        if (step + 1) % 25 == 0 or step + 1 == RULE['external_steps']:
            save_resume(folder, {'route': route, 'steps': len(rows), 'rows': rows,
                                  'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                  'manifest_sha256': digest(root / 'manifest.json')})
            print(f'EXTERNAL {route} step={step+1}/{RULE["external_steps"]} loss={rows[-1]["loss"]:.6f} update={seconds:.2f}s', flush=True)
        del result, loss, video, q, targets
    changed = S.check_changes(model, before, configuration['trainable_names'])
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Nonfinite external weights')
    final = folder / 'external-final.pth'
    atomic_torch(final, {'route': route, 'steps': len(rows), 'model': model.cpu().state_dict(),
                         'feature_sidecar': 'mamba' if route == 'A1' else 'none', 'backbone_architecture': 'base',
                         'manifest_sha256': digest(root / 'manifest.json')})
    save(folder / 'steps.json', rows)
    publish(root, done, [final, folder / 'steps.json', folder / 'initial.json'],
            frozen_tensors_exact=True, changed_tensors=changed, steps=len(rows))
    print('EXTERNAL COMPLETE ' + route, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--route', choices=('B1',), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--verify-smoke', action='store_true')
    a = p.parse_args()
    if a.verify_smoke:
        checked_plan(Path('/study'))
        if marker(Path('/study'), 'smoke/exclusion-verified.json') is None:
            raise ValueError('Replacement-label GPU smoke required')
        print('SMOKE exclusion verified', flush=True)
    elif a.smoke:
        exclusion_smoke(Path('/study'), Path('/acdc'), Path('/audit'), Path('/weights'))
    else:
        train(Path('/study'), Path('/acdc'), Path('/audit'), Path('/weights'), a.route)
