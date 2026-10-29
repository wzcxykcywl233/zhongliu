"""Scoped short-window teacher repair; original study and shape repair stay frozen."""
import argparse
import hashlib
import inspect
from pathlib import Path
import runpy
import sys
import textwrap
import types

import torch

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE.parent / 'region-supervision-v1'
SHAPE = PACKAGE.parent / 'region-output-shape-fix-v1'
sys.path.insert(0, str(ORIGINAL))
import core

BINDING = 'region-short-teacher-fix-v1.json'
BEFORE = 'num_windows = (T - S + step - 1) // step + 1'
AFTER = 'num_windows = max(1, (T - S + step - 1) // step + 1)'
TEACHER_SOURCE = {
    'cotracker.py': 'fc4f44d556840546ab3df4fa056b22e71106aa5b932cb553d00e16768a06841f',
    'cotracker3_online.py': '4c8bfce5546d85ba22c6b5292b870ad2aa0af1e0b6ba5a22e644c39bb496475e',
}


def needs_fix(slot, frames, window=16):
    return slot in (0, 1) and 0 < frames <= window // 2


def fixed_forward(model, slot):
    from cotracker.models.core.cotracker.cotracker import CoTracker2
    from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline
    cls = (CoTracker2, CoTrackerThreeOnline)[slot]
    if type(model) is not cls or model.window_len != 16:
        raise ValueError('Unexpected short-sequence teacher implementation')
    method = cls.forward
    path = Path(inspect.getsourcefile(method)).resolve()
    if core.APP.resolve() not in path.parents or path.name not in ('cotracker.py', 'cotracker3_online.py'):
        raise ValueError('Wrong installed teacher source')
    if hashlib.sha256(path.read_text(encoding='utf-8-sig').encode()).hexdigest() != TEACHER_SOURCE[path.name]:
        raise ValueError('Teacher implementation changed; this repair is version bound')
    source = textwrap.dedent(inspect.getsource(method))
    if source.count(BEFORE) != 1 or method.__code__.co_freevars:
        raise ValueError('Unrecognized teacher window calculation')
    namespace = dict(method.__globals__)
    exec(compile(source.replace(BEFORE, AFTER), str(path), 'exec'), namespace)
    return types.MethodType(namespace['forward'], model)


def restore_attribute(model, name, existed, value):
    if existed:
        setattr(model, name, value)
    else:
        delattr(model, name)


def teacher_call(original, slot, teacher_type, model, video, queries, draw, arm, pool):
    frames = int(video.shape[1])
    if not needs_fix(slot, frames, getattr(model, 'window_len', 16)):
        return original(slot, teacher_type, model, video, queries, draw, arm, pool)
    if frames < 2 or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Short repair requires a frozen teacher and at least two native frames')
    saved = {name: (name in model.__dict__, model.__dict__.get(name)) for name in ('forward', 'forward_window')}
    window = model.forward_window
    calls = []
    def count_window(*args, **kwargs):
        calls.append(1)
        return window(*args, **kwargs)
    model.forward = fixed_forward(model, slot)
    model.forward_window = count_window
    try:
        result = original(slot, teacher_type, model, video, queries, draw, arm, pool)
        p, v, c, audit = result
        if len(calls) != 1 or p.shape != (*v.shape, 2) or p.shape[1] != frames:
            raise ValueError('Short teacher must execute one window and return only native frames')
        if not all(bool(torch.isfinite(x).all()) for x in (p, v, c)):
            raise ValueError('Nonfinite repaired teacher output')
        audit = dict(audit, short_window_repair=dict(native_frames=frames, window_len=16,
            windows_executed=len(calls), original_windows=(frames-16+7)//8+1,
            native_output_frames=int(p.shape[1]), change=AFTER,
            note='Existing internal window padding only; no dataset frame/mask padding or threshold change'))
        print(f'TEACHER WINDOW REPAIRED slot={slot} native_frames={frames} windows={len(calls)} visible_gt09={float((v>.9).float().mean()):.6f}', flush=True)
        return p, v, c, audit
    finally:
        for name, (existed, value) in saved.items():
            restore_attribute(model, name, existed, value)


def bind(root):
    core.checked(root)
    value = dict(schema=1, manifest_sha256=core.sha(root/'manifest.json'),
        source={p.name:core.sha(p) for p in sorted(PACKAGE.iterdir()) if p.suffix in ('.py','.ps1','.txt')},
        shape_repair_source={p.name:core.sha(p) for p in sorted(SHAPE.iterdir()) if p.suffix in ('.py','.ps1','.txt')},
        before=BEFORE, after=AFTER, scope='Only frozen online teachers with <=8 native frames; no student/optimizer/loss/data selection change')
    path = root/BINDING
    core.freeze(path, value)
    return path


def short_row(row):
    return needs_fix(row['slot'], len(row['sampling']['native_frames']))


def validate_resume(value, binding):
    digest = core.sha(binding)
    saved = value.get('short_teacher_repair_sha256')
    if saved is not None and saved != digest:
        raise ValueError('Resume checkpoint belongs to another teacher repair')
    if saved is None and any(short_row(r) for r in value['records']):
        raise ValueError('Legacy resume contains unverified short-teacher steps')


def train_fixed(root, weights, audit, acdc, data, phase, arm, seed, max_step_seconds):
    import train as training
    binding = bind(root)
    folder = root/phase/f'train/seed_{seed}'/arm
    checkpoint = folder/'resume.pth'
    if checkpoint.exists():
        validate_resume(torch.load(checkpoint, map_location='cpu', weights_only=True), binding)
    completed = core.cached(root, folder/'verified.json')
    if completed and BINDING not in completed['files']:
        if any(short_row(r) for r in core.read(folder/'steps.json')):
            raise ValueError('Legacy completed training contains unverified short-teacher steps')
    loader, labels, receipt, atomic = training.S.load_protocol, training.labels, training.receipt, training.S.atomic_torch
    diagnostic = {}
    def load_protocol(checkout):
        protocol = loader(checkout)
        original = protocol.teacher_forward
        def forward(*a, **kw):
            result = teacher_call(original, *a, **kw)
            p, v, c, _ = result
            diagnostic.update(tracks_finite=bool(torch.isfinite(p).all()),
                visibility_finite=bool(torch.isfinite(v).all()), confidence_finite=bool(torch.isfinite(c).all()),
                visibility_min=float(v.min()), visibility_max=float(v.max()),
                visible_gt09_count=int((v>.9).sum()))
            return result
        protocol.teacher_forward = forward
        return protocol
    def checked_labels(root, plan, weights, sample, seed, step, models):
        diagnostic.clear()
        path = root/f'labels/seed_{seed}/{step:05d}.json'
        proof = core.cached(root, path)
        if proof and short_row(proof['binding']) and proof['files'].get(BINDING) != core.sha(binding):
            raise ValueError('Cached short-teacher labels lack repair provenance')
        try:
            return labels(root, plan, weights, sample, seed, step, models)
        except Exception as error:
            core.save(root/f'label-failures/seed_{seed}_{step:05d}.json', dict(
                step=step, seed=seed, sampling=sample['sampling'], domain=sample['domain'],
                draw=sample['draw'], native_frames=int(sample['video'].shape[0]),
                teacher_slot=step//2 if step<6 else int(training.np.random.default_rng(sample['draw']+41).integers(3)),
                output_diagnostic=dict(diagnostic), error=str(error), repair_sha256=core.sha(binding)))
            raise
    def bound_receipt(root, path, files, **extra):
        return receipt(root, path, list(files)+[binding], **extra)
    def bound_atomic(path, value):
        if path.name == 'resume.pth':
            value = dict(value, short_teacher_repair_sha256=core.sha(binding))
        return atomic(path, value)
    training.S.load_protocol = load_protocol
    training.labels = checked_labels
    training.receipt = bound_receipt
    training.S.atomic_torch = bound_atomic
    try:
        return training.train(root, weights, audit, acdc, data, phase, arm, seed, max_step_seconds)
    finally:
        training.S.load_protocol, training.labels = loader, labels
        training.receipt, training.S.atomic_torch = receipt, atomic


def main():
    script = sys.argv.pop(1)
    root = Path('/study')
    bind(root)
    if script == 'train.py':
        p = argparse.ArgumentParser()
        p.add_argument('--phase', choices=['smoke','pilot','full'], required=True)
        p.add_argument('--arm', choices=core.ARMS, required=True)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--max-step-seconds', type=float, default=180)
        a = p.parse_args()
        train_fixed(root, Path('/weights'), Path('/audit'), Path('/acdc'), Path('/data'),
            a.phase, a.arm, a.seed, a.max_step_seconds)
    elif script == 'infer.py':
        # Keep the previous validated region-output repair and its binding unchanged.
        runpy.run_path(str(SHAPE/'infer_fixed.py'), run_name='__main__')
    elif script == 'reports.py':
        runpy.run_path(str(ORIGINAL/'reports.py'), run_name='__main__')
    else:
        raise ValueError('Unexpected repair dispatch')


if __name__ == '__main__':
    main()
