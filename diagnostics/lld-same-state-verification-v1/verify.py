"""Independent GPU diagnostic. Never supplies a production-training gate."""
import argparse
import csv
import gc
from pathlib import Path
import shutil
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'lld-fnet-runtime-fix-v1'))
import binding as B
from engine import execute
from runtime import install
from measure import Phases, memory
from diagnose import environment
from domain import configure, Tap, DomainHead
from lld_data import sample
sys.path.insert(0, str(HERE.parent/'lld-numerical-audit-v1'))
import analyze as A
sys.path.insert(0, str(HERE))
import state as S

P, C, T = B.P, B.P.C, B.BASE.T
ARMS = B.ARMS
JOBS = ('original', 'repeat', 'lowmem', 'same_repeat', 'same_lowmem')
STEPS = tuple(range(6))
ANCHORS = (0, 1, 2)
RULE = dict(schema=1, steps=list(STEPS), same_prestate_steps=list(ANCHORS), arms=list(ARMS),
            jobs=list(JOBS), seed=0, numerical_limits=B.LIMITS,
            minimum_free_gib=3, job_timeout_seconds=3600,
            training_authorized=False, gate_changed=False,
            note='Fresh canonical original trajectory. Same-step replays restore all model/AdamW/RNG state; '
                 'drift trajectories restore step-0 only. Step-0 drift also supplies the same-state comparison. '
                 'Numerical failures are recorded, not waived. Nonfinite/identity/frozen failures stop immediately.')


def sources():
    folders = (HERE, HERE.parent/'lld-numerical-audit-v1')
    return {p.relative_to(C.CHECKOUT).as_posix(): C.sha(p) for folder in folders
            for p in sorted(folder.iterdir()) if p.suffix in ('.py', '.ps1', '.txt')}


def manifest(root, baseline, previous, audit):
    identity = dict(rule=RULE, upstream=B.identity(baseline, previous, audit), sources=sources())
    C.freeze(root/'manifest.json', identity)
    return identity


def folder(root, arm, job):
    if arm not in ARMS or job not in JOBS: raise ValueError('Unknown probe')
    return root/arm/job


def job_steps(job):
    return ANCHORS[1:] if job.startswith('same_') else STEPS


def receipt(root, arm, job):
    f = folder(root, arm, job); path = f/'complete.json'
    if not path.exists(): return None
    r = C.read(path)
    if (r.get('complete'), r.get('manifest'), r.get('arm'), r.get('job')) != (
            True, C.sha(root/'manifest.json'), arm, job):
        raise ValueError('Invalid probe receipt')
    if r.get('environment_sha256') != C.sha(root/'environment.json'):
        raise ValueError('Probe environment record changed')
    expected = {'steps.json'} | {f'step-{s}.pth' for s in job_steps(job)}
    if job == 'original': expected |= {f'pre-{s}.pth' for s in ANCHORS}
    if set(r['files']) != expected: raise ValueError('Incomplete probe receipt')
    C.verify(f, r['files'])
    rows = C.read(f/'steps.json')
    if [r['step'] for r in rows] != list(job_steps(job)): raise ValueError('Incomplete probe sequence')
    if job != 'original' and r['original_receipt'] != C.sha(folder(root, arm, 'original')/'complete.json'):
        raise ValueError('Canonical trajectory changed')
    return r


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def make_model(weights, arm, lowmem):
    model, _ = T.new_model(weights, 0, 'M0')
    configure(model, arm)
    if lowmem: install(model)
    model.cuda()
    side = [p for n, p in model.named_parameters() if n.startswith(P.PREFIX)]
    enc = [p for n, p in model.named_parameters() if n.startswith(P.ENCODER_PREFIXES)]
    opt = torch.optim.AdamW([dict(params=side, lr=P.RULE['sidecar_lr']),
                            dict(params=enc, lr=P.RULE['encoder_lr'])], weight_decay=P.RULE['weight_decay'])
    head = disc = tap = None
    if arm != 'F0':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.RULE['seed_base'] + 991)
            head = DomainHead(model.latent_dim)
        head.cuda().train()
        disc = torch.optim.AdamW(head.parameters(), lr=P.RULE['classifier_lr'], weight_decay=P.RULE['weight_decay'])
        tap = Tap(model, 'fnet')
    return model, opt, head, disc, tap, side + enc


def probe(root, baseline, previous, audit, weights, lld, data, arm, job):
    manifest(root, baseline, previous, audit)
    out = folder(root, arm, job); out.mkdir(parents=True, exist_ok=True)
    if receipt(root, arm, job):
        print(f'CACHE VERIFIED {arm}/{job}', flush=True); return
    canonical = folder(root, arm, 'original')
    if job != 'original' and not receipt(root, arm, 'original'):
        raise ValueError('Complete original trajectory required first')
    if shutil.disk_usage(root).free < RULE['minimum_free_gib']*2**30:
        raise ValueError('Need at least 3 GiB free for this diagnostic job; reserve 20 GiB for all; nothing deleted')
    C.freeze(root/'environment.json', environment())
    study = P.checked(baseline, previous)
    old = C.read(previous/'manifest.json')
    C.verify(weights, old['weights']); C.verify(data, old['files']['train-40'])
    _, reference, original_proof = P.old_training(previous, 0)
    model, opt, head, disc, tap, params = make_model(weights, arm, 'lowmem' in job)
    initial = T.S.state_hashes(model)
    train_names = [n for n, p in model.named_parameters() if p.requires_grad]
    if {n: h for n, h in initial.items() if n.startswith(P.PREFIX)} != original_proof['initial_sidecar']:
        raise ValueError('Initial sidecar differs from paired reference')
    frozen = {n: h for n, h in initial.items() if n not in train_names}
    rows = []; paths = []
    classifier = list(head.parameters()) if head is not None else []
    try:
        for step in job_steps(job):
            s = sample(study, lld, data, 0, step)
            label, row = B.BASE.input_labels(root, previous, old, weights, s, 0, step, reference[step])
            T.seed_all(s['draw'] + 47)
            # Clear stale gradients/allocator cache for every mode, not only candidate.
            opt.zero_grad(set_to_none=True)
            if disc is not None: disc.zero_grad(set_to_none=True)
            if job != 'original' and (job.startswith('same_') or step == 0):
                saved = load(canonical/f'pre-{step}.pth')
                pre_hash = S.restore(saved, model, opt, head, disc)
                del saved
            else:
                saved = S.capture(model, opt, head, disc)
                pre_hash = S.fingerprint(saved)
                if job == 'original' and step in ANCHORS:
                    path = out/f'pre-{step}.pth'; T.S.atomic_torch(path, saved); paths.append(path)
                del saved
            if job != 'original':
                cr = C.read(canonical/'steps.json')[step]
                if any(row[k] != cr[k] for k in P.PAIR_KEYS): raise ValueError('Input/label pairing changed')
                if (job.startswith('same_') or step == 0) and pre_hash != cr['prestate_sha256']:
                    raise ValueError('Restored prestate differs from canonical')
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            def progress(phases):
                C.save(out/'partial.json', dict(complete=False, arm=arm, job=job, step=step,
                                               completed_steps=[r['step'] for r in rows], **phases.report()))
                part = next(reversed(phases.seconds))
                print(f'PHASE {arm}/{job} step={step} {part}={phases.seconds[part]:.2f}s', flush=True)
            phases = Phases(progress)
            start = time.perf_counter()
            metrics, packet = execute(model, s, label, step, arm, params, classifier,
                                      opt, disc, head, tap, phases, True, evidence=True)
            wall = time.perf_counter() - start
            after = T.S.state_hashes(model)
            if any(after[n] != h for n, h in frozen.items()): raise ValueError('Frozen tensor changed')
            # Frozen weights are checked exactly above; omit their repeated payload to save disk.
            packet['student'] = {n: v for n, v in packet['student'].items() if n in train_names}
            packet['optimizer'] = S.named_optimizer(opt, S.optimizer_names(model, opt))
            packet['classifier_optimizer'] = S.named_optimizer(disc, S.optimizer_names(head, disc))
            if any(not torch.isfinite(t).all() for group in packet.values()
                   for t in (group if isinstance(group, list) else group.values())):
                raise ValueError('Nonfinite diagnostic packet')
            row.update(metrics, **phases.report(), **memory(), arm=arm, job=job,
                       prestate_sha256=pre_hash, frozen_exact=True,
                       student_state_sha256=S.fingerprint(packet['student']),
                       evidence_wall_seconds=wall, frames=len(s['video']), query_count=s['queries'].shape[1])
            path = out/f'step-{step}.pth'; T.S.atomic_torch(path, packet); paths.append(path)
            del packet
            rows.append(row); C.save(out/'steps.json', rows)
            print(f'PROBE {arm}/{job} step={step} {row["domain"]} '
                  f'compute={row["measured_seconds"]:.2f}s peak={row["peak_allocated"]/2**30:.2f}GiB', flush=True)
    finally:
        if tap: tap.close()
    paths.append(out/'steps.json')
    C.save(out/'complete.json', dict(complete=True, manifest=C.sha(root/'manifest.json'), arm=arm, job=job,
        environment_sha256=C.sha(root/'environment.json'),
        original_receipt=C.sha(canonical/'complete.json') if job != 'original' else None,
        files={p.name: C.sha(p) for p in paths}))
    print(f'PROBE COMPLETE {arm}/{job}; numerical verdict deferred to CPU report', flush=True)


def comparison(a, b, step):
    if set(a) != set(A.GROUPS) | {'optimizer', 'classifier_optimizer'} or set(a) != set(b):
        raise ValueError('Unexpected packet schema')
    groups, rows = A.compare_all({k: a[k] for k in A.GROUPS}, {k: b[k] for k in A.GROUPS}, B.LIMITS, step)
    # AdamW moments are diagnostic, not judged with a newly invented tolerance.
    for name in ('optimizer', 'classifier_optimizer'):
        if a[name].keys() != b[name].keys(): raise ValueError('Optimizer state keys changed')
        values = []
        for key in a[name]:
            v = A.tensor_stats(a[name][key], b[name][key], 0., 0.)
            v.pop('max_tolerance_ratio'); v.pop('worst_ratio_flat')
            rows.append(dict(step=step, group=name, tensor=key, **v)); values.append(v)
        groups[name] = dict(exact=all(v['exact'] for v in values),
                           max_abs_difference=max((v['max_abs_difference'] for v in values), default=0.),
                           diagnostic_only=True)
    return groups, rows


def paired_rows(a, b, require_prestate):
    if any(a[k] != b[k] for k in P.PAIR_KEYS): raise ValueError('Unpaired report inputs')
    if not a['frozen_exact'] or not b['frozen_exact']: raise ValueError('Frozen proof missing')
    if require_prestate and a['prestate_sha256'] != b['prestate_sha256']:
        raise ValueError('Same-state claim lacks exact prestate')


def runtime_summary(original, candidate):
    import math
    for rows in (original, candidate):
        if sorted(rows) != list(STEPS): raise ValueError('Incomplete runtime sequence')
        for r in rows.values():
            for k in ('measured_seconds', 'peak_allocated', 'device_total'):
                if not math.isfinite(r[k]) or r[k] <= 0: raise ValueError('Invalid runtime evidence')
    tr = [s for s in STEPS if original[s]['domain'] == 'trackrad']
    peak_ratio = max(candidate[s]['peak_allocated'] for s in tr) / max(original[s]['peak_allocated'] for s in tr)
    time_ratio = candidate[tr[-1]]['measured_seconds'] / original[tr[-1]]['measured_seconds']
    return dict(diagnostic_only=True, trackrad_peak_ratio=peak_ratio, last_trackrad_time_ratio=time_ratio,
        memory_reduction_at_least_15_percent=peak_ratio <= B.LIMITS['maximum_peak_ratio'],
        all_steps_within_180_seconds=all(r['measured_seconds'] <= B.LIMITS['training_step_seconds'] for r in candidate.values()),
        all_peaks_below_85_percent_device=all(r['peak_allocated'] <= B.LIMITS['maximum_device_fraction']*r['device_total'] for r in candidate.values()),
        last_trackrad_within_90_seconds=candidate[tr[-1]]['measured_seconds'] <= B.LIMITS['warmed_trackrad_seconds'],
        last_trackrad_no_more_than_15_percent_slower=time_ratio <= B.LIMITS['maximum_slowdown'])


def report(root, baseline, previous, audit, arms):
    torch.set_num_threads(2)
    manifest(root, baseline, previous, audit)
    allrows = []; comparisons = []; job_receipts = {}; runtimes = {}
    for arm in arms:
        data = {}
        for job in JOBS:
            if not receipt(root, arm, job): raise ValueError('Missing job: ' + arm + '/' + job)
            f = folder(root, arm, job)
            data[job] = {r['step']: r for r in C.read(f/'steps.json')}
            job_receipts[f'{arm}/{job}'] = C.sha(f/'complete.json')
        runtimes[arm] = runtime_summary(data['original'], data['lowmem'])
        for scope, steps in (('same_prestate', ANCHORS), ('trajectory_drift', STEPS)):
            for step in steps:
                a = load(folder(root, arm, 'original')/f'step-{step}.pth')
                for mode in ('repeat', 'lowmem'):
                    job = ('same_' + mode) if scope == 'same_prestate' and step else mode
                    ra, rb = data['original'][step], data[job][step]
                    paired_rows(ra, rb, scope == 'same_prestate' or step == 0)
                    b = load(folder(root, arm, job)/f'step-{step}.pth')
                    groups, rows = comparison(a, b, step)
                    allrows += [dict(arm=arm, scope=scope, mode=mode, **r) for r in rows]
                    comparisons.append(dict(arm=arm, scope=scope, step=step, mode=mode, groups=groups,
                        original_seconds=ra['measured_seconds'], candidate_seconds=rb['measured_seconds'],
                        original_peak_gib=ra['peak_allocated']/2**30, candidate_peak_gib=rb['peak_allocated']/2**30,
                        existing_tolerances_pass=all(groups[k]['within_existing_tolerance'] for k in A.GROUPS)))
                    del b
                del a
    verdicts = {}
    for arm in arms:
        selected = [r for r in comparisons if r['arm'] == arm]
        verdicts[arm] = {f'{scope}_{mode}_existing_tolerances_pass': all(r['existing_tolerances_pass'] for r in selected
            if r['scope'] == scope and r['mode'] == mode)
            for scope in ('same_prestate', 'trajectory_drift') for mode in ('repeat', 'lowmem')}
    zero_control = None
    if 'F0' in arms and 'Fprobe' in arms:
        zero_control = {job: [r['student_state_sha256'] for r in C.read(folder(root, 'F0', job)/'steps.json')] ==
                            [r['student_state_sha256'] for r in C.read(folder(root, 'Fprobe', job)/'steps.json')]
                        for job in JOBS}
    summary = dict(complete=True, training_authorized=False, gate_changed=False, arms=arms,
        same_prestate_steps=list(ANCHORS), drift_steps=list(STEPS), limits_unchanged=B.LIMITS,
        verdicts=verdicts, runtime_diagnostics=runtimes, zero_control_student_exact=zero_control, comparisons=comparisons,
        job_receipts=job_receipts, manifest_sha256=C.sha(root/'manifest.json'),
        note='Completion is not acceptance. Original-repeat noise is descriptive, not a relaxed gate. '
             'Six updates do not establish long-run stability or task accuracy. Optimizer moments diagnostic only. '
             'No production training or validation/test inference was run.')
    reportdir = root/'reports'/('all' if len(arms) == len(ARMS) else 'F0')
    C.save(reportdir/'summary.json', summary)
    with (reportdir/'tensor-differences.csv').open('w', newline='', encoding='utf-8') as f:
        fields = list(dict.fromkeys(k for row in allrows for k in row))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(allrows)
    compact = []
    for r in comparisons:
        for group in A.GROUPS:
            g = r['groups'][group]
            compact.append(dict(arm=r['arm'], scope=r['scope'], step=r['step'], mode=r['mode'], group=group,
                                **g))
    with (reportdir/'comparisons.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(compact[0])); w.writeheader(); w.writerows(compact)
    C.save(root/'summary.json', {k: v for k, v in summary.items() if k != 'comparisons'})
    for arm, v in verdicts.items(): print('RESULT ' + arm + ': ' + str(v), flush=True)
    print('REPORT COMPLETE /diagnostic/summary.json; training_authorized=false; old gate unchanged', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['probe', 'report'])
    p.add_argument('--arm', choices=ARMS); p.add_argument('--job', choices=JOBS)
    p.add_argument('--scope', choices=['F0', 'all'], default='all')
    a = p.parse_args()
    paths = [Path(x) for x in ('/diagnostic', '/baseline', '/previous', '/audit')]
    if a.command == 'report': report(*paths, list(ARMS) if a.scope == 'all' else ['F0'])
    else: probe(*paths, Path('/weights'), Path('/lld'), Path('/data'), a.arm, a.job)


if __name__ == '__main__': main()
