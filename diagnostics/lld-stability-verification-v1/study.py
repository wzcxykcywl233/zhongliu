"""Graph isolation, paired 100-step low-memory runs, fixed train-only probes."""
import argparse
import csv
import gc
import time
import torch
import support as H

V, B, P, C, T, S = H.V, H.B, H.P, H.C, H.T, H.S


def inputs(previous, baseline, weights, data):
    study = P.checked(baseline, previous); old = C.read(previous/'manifest.json')
    C.verify(weights, old['weights']); C.verify(data, old['files']['train-40'])
    _, reference, proof = P.old_training(previous, 0)
    return study, old, reference, proof


def get_sample(root, previous, weights, lld, data, state, step):
    study, old, reference, _ = state
    s = V.sample(study, lld, data, 0, step)
    label, row = B.BASE.input_labels(root, previous, old, weights, s, 0, step, reference[step])
    return s, label, row


def identity(model, proof):
    initial = T.S.state_hashes(model)
    if {n: h for n, h in initial.items() if n.startswith(P.PREFIX)} != proof['initial_sidecar']:
        raise ValueError('Fresh student initialization changed')
    active = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = {n: h for n, h in initial.items() if n not in active}
    return initial, active, frozen


def frozen_check(model, frozen):
    now = T.S.state_hashes(model)
    if any(now[n] != h for n, h in frozen.items()): raise ValueError('Frozen tensors changed')
    return now


def graph(root, evidence, baseline, previous, audit, weights, lld, data):
    H.plan(root, evidence, baseline, previous, audit); H.check_environment(root, evidence)
    out = root/'graph'; out.mkdir(parents=True, exist_ok=True)
    if H.completed(root, out, 'graph'):
        print('CACHE VERIFIED graph isolation', flush=True); return
    state = inputs(previous, baseline, weights, data)
    model, opt, head, disc, tap, _ = V.make_model(weights, 'Fprobe', True)
    initial, _, _ = identity(model, state[3]); head_initial = S.fingerprint(dict(head.state_dict()))
    rows = []
    try:
        for step in (0, 1):
            s, label, row = get_sample(root, previous, weights, lld, data, state, step)
            T.seed_all(s['draw']+47); tap.reset(len(s['video']))
            start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            video, q = s['video'][None].cuda(), s['queries'].cuda()
            output = model(video=video, queries=q, iters=C.RULE['iterations'], is_train=True)
            task = T.S.training_loss(output, {k: label[k].cuda() for k in ('tracks', 'visibility')})
            captures = len(tap.values)
            proof = H.graph_check(model, head, tap.embedding(), task, s['domain'])
            if len(tap.values) != captures: raise ValueError('Checkpoint replay duplicated domain hook')
            if any(p.grad is not None for p in list(model.parameters())+list(head.parameters())):
                raise ValueError('Graph-only check wrote gradient buffers')
            if T.S.state_hashes(model) != initial or S.fingerprint(dict(head.state_dict())) != head_initial:
                raise ValueError('Graph-only check changed weights or buffers')
            if opt.state or disc.state: raise ValueError('Graph-only check updated AdamW')
            torch.cuda.synchronize()
            rows.append(dict(row, **proof, seconds=time.perf_counter()-start, **V.memory()))
            del output, task, video, q
            tap.values.clear(); gc.collect(); torch.cuda.empty_cache()
            C.save(out/'partial.json', dict(complete=False, rows=rows))
            print(f'GRAPH PASSED sample={step}; zero student gradient paths; positive GRL/task gradients present', flush=True)
    finally:
        tap.close()
    C.save(out/'summary.json', dict(complete=True, weights_unchanged=True, optimizer_unchanged=True, rows=rows))
    H.publish(root, out, ['summary.json'], 'graph')


def fixed_probe(model, tap, s, label, initial=None):
    """Fixed training-pool diagnostic, not DSC or unseen-dataset evaluation."""
    rng = S.rng_state(); modes = {n: m.training for n, m in model.named_modules()}
    try:
        T.seed_all(s['draw']+47)
        if tap: tap.reset(len(s['video']))
        with torch.no_grad():
            out = model(video=s['video'][None].cuda(), queries=s['queries'].cuda(),
                        iters=C.RULE['iterations'], is_train=True)
            loss = T.S.training_loss(out, {k: label[k].cuda() for k in ('tracks', 'visibility')})
            packet = [v.detach().cpu().clone() for v in out[:3]]
        if not torch.isfinite(loss) or any(not torch.isfinite(x).all() for x in packet):
            raise ValueError('Nonfinite fixed-probe output')
        tracks, vis, conf = packet
        w, h = s['video'].shape[-1], s['video'].shape[-2]
        oob = ((tracks[..., 0] < 0) | (tracks[..., 0] >= w) | (tracks[..., 1] < 0) | (tracks[..., 1] >= h))
        result = dict(point_loss=float(loss), visibility_mean=float(vis.mean()), confidence_mean=float(conf.mean()),
                      out_of_bounds_fraction=float(oob.float().mean()))
        if initial is not None:
            diff = tracks[:, 1:]-initial[0][:, 1:]
            result.update(coordinate_shift_rms_px=float(diff.square().sum(-1).mean().sqrt()),
                          visibility_mean_abs_change=float((vis-initial[1]).abs().mean()),
                          confidence_mean_abs_change=float((conf-initial[2]).abs().mean()))
        else:
            result.update(coordinate_shift_rms_px=0., visibility_mean_abs_change=0., confidence_mean_abs_change=0.)
        return result, packet
    finally:
        if tap: tap.values.clear()
        S.restore_rng(rng)
        if modes != {n: m.training for n, m in model.named_modules()}: raise ValueError('Probe changed model modes')


def check_probes(root, out, previous, weights, lld, data, state, model, tap, count, baseline_packets, probes):
    before = T.S.state_hashes(model)
    for index in H.RULE['fixed_probe_sample_indices']:
        s, label, binding = get_sample(root, previous, weights, lld, data, state, index)
        result, packet = fixed_probe(model, tap, s, label, baseline_packets.get(index))
        if count == 0: baseline_packets[index] = packet
        probes.append(dict(after_updates=count, sample_index=index, **binding, **result))
        del packet
    if before != T.S.state_hashes(model): raise ValueError('Fixed probes changed weights/buffers')
    C.save(out/'probes.json', probes)
    print(f'PROBES COMPLETE {out.name} after={count}; four fixed train-only clips', flush=True)


def train(root, evidence, baseline, previous, audit, weights, lld, data, arm):
    if arm not in H.ARMS: raise ValueError('Unknown arm')
    H.plan(root, evidence, baseline, previous, audit); H.check_environment(root, evidence)
    if not H.completed(root, root/'graph', 'graph'): raise ValueError('Graph isolation check required')
    out = root/'train'/arm; out.mkdir(parents=True, exist_ok=True)
    if H.completed(root, out, arm):
        print(f'CACHE VERIFIED 100-step {arm}', flush=True); return
    if (out/'failure.json').exists():
        raise ValueError('Prior stability failure requires review; cannot skip it on resume')
    state = inputs(previous, baseline, weights, data)
    model, opt, head, disc, tap, params = V.make_model(weights, arm, True)
    initial, active, frozen = identity(model, state[3])
    rows, probes, baseline_packets = [], [], {}
    checkpoint = out/'resume.pth'
    def save_checkpoint():
        T.S.atomic_torch(checkpoint, dict(plan=C.sha(root/'plan.json'), arm=arm,
            environment=C.sha(root/'environment.json'), state=S.capture(model, opt, head, disc),
            initial=initial, rows=rows, probes=probes, baseline_packets=baseline_packets))
    try:
        if checkpoint.exists():
            c = H.load(checkpoint)
            if (c['plan'], c['arm'], c['environment'], c['initial']) != (
                    C.sha(root/'plan.json'), arm, C.sha(root/'environment.json'), initial):
                raise ValueError('Resume identity changed')
            H.validate_records(c['rows'])
            H.validate_probes(c['probes'], len(c['rows']))
            for row in c['rows']:
                bound = C.read(root/f'inputs/seed_0/{row["step"]:05d}.json')
                if any(row[k] != bound[k] for k in P.PAIR_KEYS): raise ValueError('Resume input pairing changed')
            S.restore(c['state'], model, opt, head, disc)
            rows, probes, baseline_packets = c['rows'], c['probes'], c['baseline_packets']
            del c; frozen_check(model, frozen)
            if any(not torch.isfinite(p).all() for p in params): raise ValueError('Nonfinite restored student')
            if head is not None and any(not torch.isfinite(p).all() for p in head.parameters()):
                raise ValueError('Nonfinite restored classifier')
            if set(baseline_packets) != set(H.RULE['fixed_probe_sample_indices']): raise ValueError('Missing fixed-probe baseline')
            print(f'RESUME {arm} completed_updates={len(rows)}', flush=True)
        else:
            check_probes(root, out, previous, weights, lld, data, state, model, tap, 0, baseline_packets, probes)
            save_checkpoint()
        for step in range(len(rows), H.RULE['updates']):
            torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            phases = V.Phases()
            with phases.part('data_and_labels'):
                s, label, row = get_sample(root, previous, weights, lld, data, state, step)
            metrics, _ = V.execute(model, s, label, step, arm, params, list(head.parameters()) if head else [],
                opt, disc, head, tap, phases, step in H.RULE['audit_steps'], evidence=False)
            torch.cuda.synchronize()
            row.update(metrics, **phases.report(), **V.memory(), seconds=time.perf_counter()-start,
                       frames=len(s['video']), query_count=s['queries'].shape[1])
            rows.append(row); C.save(out/'steps.json', rows)
            try:
                H.validate_records(rows)
                if any(not torch.isfinite(p).all() for p in params): raise ValueError('Nonfinite updated student')
                if head is not None and any(not torch.isfinite(p).all() for p in head.parameters()):
                    raise ValueError('Nonfinite updated classifier')
            except Exception as exc:
                C.save(out/'failure.json', dict(step=step, error=str(exc), review_required=True))
                save_checkpoint(); raise
            count = step+1
            if count in H.RULE['probe_after_updates']:
                check_probes(root, out, previous, weights, lld, data, state, model, tap, count, baseline_packets, probes)
            if count%H.RULE['checkpoint_every'] == 0 or count in H.RULE['probe_after_updates']:
                frozen_check(model, frozen); save_checkpoint()
                print(f'TRAIN {arm} {count}/100 point={row["point"]:.6f} seconds={row["seconds"]:.2f} '
                      f'peak={row["peak_allocated"]/2**30:.2f}GiB', flush=True)
        final = frozen_check(model, frozen)
        if not any(final[n] != initial[n] for n in active if n.startswith(P.PREFIX)):
            raise ValueError('Sidecar did not update')
        if not any(final[n] != initial[n] for n in active if n.startswith(P.ENCODER_PREFIXES)):
            raise ValueError('Partial encoder did not update')
        C.save(out/'proof.json', dict(complete=True, arm=arm, updates=len(rows), frozen_exact=True,
            initial_state_sha256=initial, final_state_sha256=final, trainable=active,
            sidecar_updated=True, partial_encoder_updated=True, full_training_authorized=False))
        H.publish(root, out, ['resume.pth', 'steps.json', 'probes.json', 'proof.json'], arm)
        print(f'TRAIN COMPLETE {arm}; no automatic pilot/full', flush=True)
    except (ValueError, FloatingPointError, AssertionError) as exc:
        C.save(out/'failure.json', dict(completed_updates=len(rows), error=str(exc), review_required=True))
        raise
    finally:
        if tap: tap.close()


def report(root, evidence, baseline, previous, audit):
    H.plan(root, evidence, baseline, previous, audit)
    if not H.completed(root, root/'graph', 'graph'): raise ValueError('Graph check missing')
    results, all_probes, all_rows = {}, [], []
    reference = None; initial_reference = None
    for arm in H.ARMS:
        out = root/'train'/arm
        if not H.completed(root, out, arm): raise ValueError('Incomplete arm: '+arm)
        rows = C.read(out/'steps.json'); probes = C.read(out/'probes.json')
        proof = C.read(out/'proof.json')
        if (proof['complete'], proof['arm'], proof['updates'], proof['frozen_exact'],
            proof['sidecar_updated'], proof['partial_encoder_updated'], proof['full_training_authorized']) != (
                True, arm, H.RULE['updates'], True, True, True, False):
            raise ValueError('Invalid training proof')
        if initial_reference is None: initial_reference = proof['initial_state_sha256']
        elif proof['initial_state_sha256'] != initial_reference: raise ValueError('Initial students differ across arms')
        H.validate_records(rows)
        if len(rows) != H.RULE['updates']: raise ValueError('Incomplete training')
        H.validate_probes(probes, H.RULE['updates'])
        binding = [{k: r[k] for k in P.PAIR_KEYS} for r in rows]
        if reference is None: reference = binding
        elif binding != reference: raise ValueError('Training inputs/labels are not paired across arms')
        alerts = []
        for p in probes:
            bound = C.read(root/f'inputs/seed_0/{p["sample_index"]:05d}.json')
            if any(p[k] != bound[k] for k in P.PAIR_KEYS): raise ValueError('Fixed-probe input changed')
            a = next(r for r in probes if r['after_updates'] == 0 and r['sample_index'] == p['sample_index'])
            if H.loss_alert(a['point_loss'], p['point_loss']):
                alerts.append(dict(after_updates=p['after_updates'], sample_index=p['sample_index'],
                    initial_loss=a['point_loss'], current_loss=p['point_loss']))
            all_probes.append(dict(arm=arm, **p))
        timing = {}
        for domain in ('trackrad', 'lld_synthetic'):
            import statistics
            s = [r for r in rows if r['domain'] == domain]
            if len(s) != 50: raise ValueError('Mixed-domain step count changed')
            timing[domain] = dict(steps=len(s), median_seconds=statistics.median(r['seconds'] for r in s),
                                 max_seconds=max(r['seconds'] for r in s), peak_gib=max(r['peak_allocated'] for r in s)/2**30)
        results[arm] = dict(technical_checks_passed=True, frozen_exact=True, updates=100, runtime=timing,
            fixed_probe_loss_review_alerts=alerts, needs_loss_review=bool(alerts),
            final_probes=[{k: p[k] for k in ('sample_index', 'domain', 'point_loss', 'coordinate_shift_rms_px',
                'out_of_bounds_fraction', 'visibility_mean', 'confidence_mean')} for p in probes if p['after_updates'] == 100])
        all_rows += [dict(arm=arm, **r) for r in rows]
    C.save(root/'summary.json', dict(complete=True, graph_isolation_passed=True, paired_inputs_and_labels=True,
        arms=results, previous_strict_gate_changed=False, numerical_equivalence_claimed=False,
        full_training_authorized=False, validation_or_test_accessed=False,
        note='Bounded stability only; fixed clips are sampled from training pools, not validation. '
             'TrackRAD losses compare teacher pseudo-labels; LLD labels are synthetic geometry. '
             'No DSC, generalization or clinical benefit claim. Review loss alerts before deciding next stage.'))
    for name, records in (('fixed-probes.csv', all_probes), ('training-steps.csv', all_rows)):
        with (root/name).open('w', newline='', encoding='utf-8') as f:
            fields = list(dict.fromkeys(k for r in records for k in r))
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    for arm, r in results.items():
        print(f'RESULT {arm}: 100 updates; frozen exact; fixed-loss alerts={len(r["fixed_probe_loss_review_alerts"])}; '
              f'TrackRAD median={r["runtime"]["trackrad"]["median_seconds"]:.2f}s', flush=True)
    print('REPORT COMPLETE /study/summary.json; no automatic next stage', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['plan', 'graph', 'train', 'report'])
    p.add_argument('--arm', choices=H.ARMS); a = p.parse_args()
    paths = [H.Path(x) for x in ('/study', '/evidence', '/baseline', '/previous', '/audit')]
    data = [H.Path(x) for x in ('/weights', '/lld', '/data')]
    if a.command == 'plan':
        H.plan(*paths, deep=True); print('PLAN VERIFIED 100 steps x four F arms; old strict gate unchanged', flush=True)
    elif a.command == 'graph': graph(*paths, *data)
    elif a.command == 'train': train(*paths, *data, a.arm)
    else: report(*paths)


if __name__ == '__main__': main()
