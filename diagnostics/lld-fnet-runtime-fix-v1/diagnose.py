"""Bounded real GPU comparisons; no validation/test access or production training."""
import argparse
import csv
import math
import time
import torch
import binding as B
from domain import configure, Tap, DomainHead
from lld_data import sample
from runtime import install
from measure import Phases, memory
from engine import execute

P, C, BASE = B.P, B.P.C, B.BASE


def environment():
    d = torch.cuda.get_device_properties(0)
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=d.name,
                total_memory=d.total_memory, capability=list(torch.cuda.get_device_capability()),
                matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)


def compare(reference, candidate):
    """Predeclared tolerances, checked elementwise; never silently widen them."""
    stats = {}
    for name in ('outputs', 'gradients', 'classifier_gradients', 'student', 'classifier'):
        a, b = reference[name], candidate[name]
        if isinstance(a, list):
            a, b = dict(enumerate(a)), dict(enumerate(b))
        if a.keys() != b.keys():
            raise ValueError('Evidence keys differ: ' + name)
        prefix = 'output' if name == 'outputs' else ('grad' if 'gradient' in name else 'state')
        maximum = 0.
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=B.LIMITS[prefix+'_atol'],
                                       rtol=B.LIMITS[prefix+'_rtol'], msg=lambda msg: f'{name}/{key}: {msg}')
            if a[key].numel():
                maximum = max(maximum, float((a[key].double() - b[key].double()).abs().max()))
        stats[name+'_max_abs_difference'] = maximum
    return stats


def probe(root, baseline, previous, audit, weights, lld, data, mode, arm):
    B.checked_manifest(root, baseline, previous, audit)
    folder = B.job(root, mode, arm)
    folder.mkdir(parents=True, exist_ok=True)
    if B.verified_job(root, mode, arm):
        print(f'PROFILE CACHE VERIFIED {mode}/{arm}', flush=True)
        return
    if mode == 'lowmem' and not B.verified_job(root, 'original', arm):
        raise ValueError('Original evidence required first')
    C.freeze(root/'environment.json', environment())
    study = P.checked(baseline, previous)
    old = C.read(previous/'manifest.json')
    C.verify(weights, old['weights']); C.verify(data, old['files']['train-40'])
    _, reference, proof = P.old_training(previous, 0)
    model, _ = BASE.T.new_model(weights, 0, 'M0')
    configure(model, arm)
    initial = BASE.T.S.state_hashes(model)
    if {n: h for n, h in initial.items() if n.startswith(P.PREFIX)} != proof['initial_sidecar']:
        raise ValueError('Initial sidecar changed')
    frozen = {n: h for n, h in initial.items() if not dict(model.named_parameters()).get(n, torch.empty(0)).requires_grad}
    if mode == 'lowmem':
        install(model)
    model.cuda()
    side = [p for n, p in model.named_parameters() if n.startswith(P.PREFIX)]
    enc = [p for n, p in model.named_parameters() if n.startswith(P.ENCODER_PREFIXES)]
    params = side + enc
    opt = torch.optim.AdamW([dict(params=side, lr=P.RULE['sidecar_lr']), dict(params=enc, lr=P.RULE['encoder_lr'])],
                            weight_decay=P.RULE['weight_decay'])
    head = tap = disc_opt = None
    classifier = []
    if arm != 'F0':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(C.RULE['seed_base'] + 991)
            head = DomainHead(model.latent_dim)
        head.cuda().train(); classifier = list(head.parameters())
        disc_opt = torch.optim.AdamW(classifier, lr=P.RULE['classifier_lr'], weight_decay=P.RULE['weight_decay'])
        tap = Tap(model, 'fnet')
    rows = []
    try:
        for step in B.STEPS:
            torch.cuda.reset_peak_memory_stats()
            def progress(phases):
                C.save(folder/'partial.json', dict(complete=False, mode=mode, arm=arm, step=step,
                                                  completed_steps=rows, **phases.report()))
                part = next(reversed(phases.seconds))
                print(f'PROFILE {mode}/{arm} step={step} {part}={phases.seconds[part]:.2f}s '
                      f'peak={memory()["peak_allocated"]/2**30:.2f}GiB', flush=True)
            phases = Phases(progress)
            with phases.part('data_read_and_sampling'):
                s = sample(study, lld, data, 0, step)
            with phases.part('label_binding'):
                label, row = BASE.input_labels(root, previous, old, weights, s, 0, step, reference[step])
            row.update(mode=mode, arm=arm, frames=len(s['video']), queries=s['queries'].shape[1])
            metrics, packet = execute(model, s, label, step, arm, params, classifier, opt, disc_opt,
                                      head, tap, phases, True, evidence=True)
            row.update(metrics, **phases.report(), **memory())
            if mode == 'lowmem':
                ref = torch.load(B.job(root, 'original', arm)/f'step-{step}.pth', map_location='cpu', weights_only=True)
                try:
                    row['equivalence'] = compare(ref, packet)
                except (AssertionError, ValueError) as exc:
                    C.save(folder/'equivalence-failure.json',dict(complete=False,mode=mode,arm=arm,
                        step=step,error=str(exc)[:8000],runtime_record=row))
                    BASE.T.S.atomic_torch(folder/'failed-evidence.pth',packet)
                    raise
                del ref
            after = BASE.T.S.state_hashes(model)
            if any(after[n] != h for n, h in frozen.items()):
                raise ValueError('Frozen tensors changed in runtime probe')
            row['student_state_sha256'] = after
            start = time.perf_counter()
            BASE.T.S.atomic_torch(folder/f'step-{step}.pth', packet)
            row['evidence_io_seconds'] = time.perf_counter() - start
            del packet
            rows.append(row)
            C.save(folder/'steps.json', rows)
    finally:
        if tap: tap.close()
    files = [folder/'steps.json'] + [folder/f'step-{s}.pth' for s in B.STEPS]
    C.save(folder/'verified.json', dict(complete=True, manifest=C.sha(root/'manifest.json'),
                                      arm=arm, mode=mode, files={p.name: C.sha(p) for p in files}))
    print(f'PROFILE COMPLETE {mode}/{arm}', flush=True)


def assess(original, lowmem):
    reasons = []
    a = [r for r in original if r['domain'] == 'trackrad']
    b = [r for r in lowmem if r['domain'] == 'trackrad']
    if [r['step'] for r in original] != list(B.STEPS) or [r['step'] for r in lowmem] != list(B.STEPS):
        raise ValueError('Incomplete probe steps')
    for r in original + lowmem:
        for key in ('peak_allocated','device_total','measured_seconds'):
            if not math.isfinite(r[key]) or r[key] <= 0:
                raise ValueError('Invalid timing/memory evidence: '+key)
    if any(not r.get('equivalence') for r in lowmem): reasons.append('missing_numerical_equivalence')
    ratio = max(r['peak_allocated'] for r in b) / max(r['peak_allocated'] for r in a)
    if ratio > B.LIMITS['maximum_peak_ratio']: reasons.append('memory_reduction_less_than_15_percent')
    if any(r['peak_allocated'] > B.LIMITS['maximum_device_fraction'] * r['device_total'] for r in lowmem):
        reasons.append('insufficient_device_memory_margin')
    if any(r['measured_seconds'] > B.LIMITS['training_step_seconds'] for r in lowmem):
        reasons.append('training_step_exceeds_180_seconds')
    if b[-1]['measured_seconds'] > B.LIMITS['warmed_trackrad_seconds']:
        reasons.append('warmed_trackrad_step_exceeds_90_seconds')
    speed_ratio = b[-1]['measured_seconds'] / a[-1]['measured_seconds']
    if speed_ratio > B.LIMITS['maximum_slowdown']: reasons.append('warmed_step_more_than_15_percent_slower')
    return dict(passed=not reasons, reasons=reasons, peak_ratio=ratio, warmed_time_ratio=speed_ratio,
                original_peak_gib=max(r['peak_allocated'] for r in a)/2**30,
                lowmem_peak_gib=max(r['peak_allocated'] for r in b)/2**30,
                original_warmed_seconds=a[-1]['measured_seconds'], lowmem_warmed_seconds=b[-1]['measured_seconds'])


def report(root, baseline, previous, audit):
    B.checked_manifest(root, baseline, previous, audit)
    files = [root/'environment.json']
    results, allrows, states = {}, [], {}
    for arm in B.ARMS:
        rows = {}
        for mode in B.MODES:
            if not B.verified_job(root, mode, arm): raise ValueError('Missing diagnostic evidence')
            folder = B.job(root, mode, arm)
            rows[mode] = C.read(folder/'steps.json')
            files.append(folder/'verified.json')
            for r in rows[mode]:
                allrows.append(dict(arm=arm, mode=mode, step=r['step'], domain=r['domain'],
                                    frames=r['frames'], **r['phase_seconds'], measured_seconds=r['measured_seconds'],
                                    peak_allocated_gib=r['peak_allocated']/2**30, peak_reserved_gib=r['peak_reserved']/2**30,
                                    device_free_gib=r['device_free']/2**30, evidence_io_seconds=r['evidence_io_seconds']))
        results[arm] = assess(rows['original'], rows['lowmem'])
        states[arm] = [r['student_state_sha256'] for r in rows['lowmem']]
    zero_exact = states['F0'] == states['Fprobe']
    summary = dict(complete=True, passed=all(r['passed'] for r in results.values()) and zero_exact,
                   comparisons=results, lowmem_zero_control_exact=zero_exact, limits=B.LIMITS,
                   note='Only runtime/gradient probes; not model accuracy. Fresh queue required. Existing results untouched.')
    C.save(root/'summary.json', summary)
    with (root/'phases.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = list(dict.fromkeys(k for row in allrows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(allrows)
    files += [root/'summary.json', root/'phases.csv']
    C.save(root/'verified.json', dict(complete=True, manifest=C.sha(root/'manifest.json'),
                                    files={p.relative_to(root).as_posix(): C.sha(p) for p in files}))
    print('RUNTIME GATE ' + ('PASSED' if summary['passed'] else 'CLOSED'), flush=True)
    for arm, r in results.items():
        print(f'PROFILE RESULT {arm}: peak {r["original_peak_gib"]:.2f}->{r["lowmem_peak_gib"]:.2f}GiB; '
              f'warmed step {r["original_warmed_seconds"]:.2f}->{r["lowmem_warmed_seconds"]:.2f}s; {r["reasons"]}', flush=True)
    if not summary['passed']:
        raise ValueError('Runtime gate closed; no large training allowed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['probe', 'report'])
    p.add_argument('--mode', choices=B.MODES); p.add_argument('--arm', choices=B.ARMS)
    a = p.parse_args()
    paths = [B.Path(x) for x in ('/diagnostic', '/baseline', '/previous', '/audit')]
    if a.command == 'report': report(*paths)
    else: probe(*paths, B.Path('/weights'), B.Path('/lld'), B.Path('/data'), a.mode, a.arm)


if __name__ == '__main__':
    main()
