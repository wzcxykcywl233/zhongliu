import copy
from contextlib import nullcontext, ExitStack
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import verify as V
import state as S

torch.set_num_threads(2)


def toy():
    torch.manual_seed(76)
    m = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    m[0].weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-6)
    return m, opt


def update(m, opt):
    opt.zero_grad(set_to_none=True)
    m(torch.ones(1, 2)).square().sum().backward(); opt.step()


class StateTests(unittest.TestCase):
    def test_weights_moments_rng_restored_exactly_without_aliases(self):
        m, opt = toy(); update(m, opt)
        saved = S.capture(m, opt); digest = S.fingerprint(saved)
        random_expected = (random.random(), np.random.rand(), torch.rand(3))
        update(m, opt); update(m, opt)
        self.assertNotEqual(S.fingerprint(S.capture(m, opt)), digest)
        self.assertEqual(S.restore(saved, m, opt), digest)
        self.assertEqual(random.random(), random_expected[0])
        self.assertEqual(np.random.rand(), random_expected[1])
        self.assertTrue(torch.equal(torch.rand(3), random_expected[2]))
        update(m, opt)
        self.assertEqual(S.fingerprint(saved), digest, 'optimizer load must not alias saved CPU moments')
        self.assertTrue(all(p.grad is None for p in m.parameters() if not p.requires_grad))

    def test_prestate_with_classifier_serialization_weights_only(self):
        m, opt = toy(); h = torch.nn.Linear(1, 1); disc = torch.optim.AdamW(h.parameters())
        update(m, opt); h(torch.ones(1, 1)).sum().backward(); disc.step()
        saved = S.capture(m, opt, h, disc)
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            path = Path(d)/'snapshot.pth'; torch.save(saved, path)
            restored = torch.load(path, map_location='cpu', weights_only=True)
        S.restore(restored, m, opt, h, disc)
        self.assertEqual(S.fingerprint(restored), S.fingerprint(S.capture(m, opt, h, disc)))
        self.assertIn('weight/exp_avg_sq', S.named_optimizer(disc, S.optimizer_names(h, disc)))

    def test_wrong_parameter_order_and_modes_refused(self):
        m, opt = toy(); saved = S.capture(m, opt)
        wrong = copy.deepcopy(saved); wrong['optimizer_names'][0].reverse()
        with self.assertRaises(ValueError): S.restore(wrong, m, opt)
        m.eval()
        with self.assertRaises(ValueError): S.restore(saved, m, opt)
        m.train(); m[0].weight.requires_grad_(True)
        with self.assertRaises(ValueError): S.restore(saved, m, opt)

    def test_fingerprint_is_order_independent_but_type_sensitive(self):
        self.assertEqual(S.fingerprint({'a': 1, 'b': 2}), S.fingerprint({'b': 2, 'a': 1}))
        self.assertNotEqual(S.fingerprint([1]), S.fingerprint((1,)))
        self.assertNotEqual(S.fingerprint(torch.ones(2)), S.fingerprint(torch.ones(2).double()))
        with self.assertRaises(TypeError): S.clone(object())


def packet(delta=0.):
    return dict(outputs=[torch.ones(3)], gradients={'weight': torch.ones(2)*1e-6},
                classifier_gradients={}, student={'weight': torch.tensor([.0268+delta, .01])}, classifier={},
                optimizer={'weight/exp_avg': torch.ones(2)*1e-9}, classifier_optimizer={})


class EvidenceTests(unittest.TestCase):
    def test_old_tolerance_failure_retained_moments_not_a_gate(self):
        a = packet(); b = packet(2.5e-6)
        b['optimizer']['weight/exp_avg'][0] *= 10
        groups, rows = V.comparison(a, b, 1)
        self.assertEqual(groups['student']['mismatched_elements'], 1)
        self.assertFalse(groups['student']['within_existing_tolerance'])
        self.assertTrue(groups['optimizer']['diagnostic_only'])
        self.assertNotIn('within_existing_tolerance', groups['optimizer'])
        self.assertTrue(any(r['group'] == 'optimizer' for r in rows))
        b['gradients']['weight'][0] = float('nan')
        with self.assertRaises(ValueError): V.comparison(a, b, 1)

    def fixture(self, root):
        V.C.save(root/'manifest.json', {'fixture': 1})
        V.C.save(root/'environment.json', {'device': 'fixture'})
        for job in V.JOBS:
            f = V.folder(root, 'F0', job); f.mkdir(parents=True)
            rows = []; paths = []
            for step in V.job_steps(job):
                row = {k: ('paired' if k != 'step' else step) for k in V.P.PAIR_KEYS}
                row.update(prestate_sha256='state'+str(step), frozen_exact=True,
                           measured_seconds=2., peak_allocated=1000, device_total=100000,
                           student_state_sha256='student', domain='trackrad' if step%2 == 0 else 'lld_synthetic')
                if job in ('repeat', 'lowmem') and step:
                    row['prestate_sha256'] += '-trajectory'
                rows.append(row)
                path = f/f'step-{step}.pth'
                torch.save(packet(2.5e-6 if job in ('lowmem', 'same_lowmem') and step else 0), path)
                paths.append(path)
                if job == 'original' and step in V.ANCHORS:
                    path = f/f'pre-{step}.pth'; torch.save({}, path); paths.append(path)
            V.C.save(f/'steps.json', rows); paths.append(f/'steps.json')
            V.C.save(f/'complete.json', dict(complete=True, manifest=V.C.sha(root/'manifest.json'), arm='F0', job=job,
                environment_sha256=V.C.sha(root/'environment.json'),
                original_receipt=V.C.sha(V.folder(root, 'F0', 'original')/'complete.json') if job != 'original' else None,
                files={p.name: V.C.sha(p) for p in paths}))

    def test_complete_report_does_not_promote_failed_candidate(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root = Path(d); self.fixture(root)
            before = {p.relative_to(root).as_posix(): V.C.sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(V, 'manifest'):
                V.report(root, root, root, root, ['F0'])
            summary = V.C.read(root/'summary.json')
            self.assertTrue(summary['complete'])
            self.assertFalse(summary['training_authorized']); self.assertFalse(summary['gate_changed'])
            self.assertFalse(summary['verdicts']['F0']['same_prestate_lowmem_existing_tolerances_pass'])
            self.assertTrue(summary['verdicts']['F0']['same_prestate_repeat_existing_tolerances_pass'])
            self.assertFalse((root/'verified.json').exists())
            V.C.verify(root, before)
            r = V.C.read(root/'reports/F0/summary.json')
            self.assertEqual(len(r['comparisons']), 18)

    def test_receipt_tampering_and_incomplete_job_refused(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root = Path(d); self.fixture(root)
            self.assertTrue(V.receipt(root, 'F0', 'repeat'))
            torch.save(packet(.02), V.folder(root, 'F0', 'repeat')/'step-1.pth')
            with self.assertRaises(ValueError): V.receipt(root, 'F0', 'repeat')
            f = V.folder(root, 'F0', 'same_lowmem'); r = V.C.read(f/'complete.json')
            del r['files']['step-1.pth']; V.C.save(f/'complete.json', r)
            with self.assertRaises(ValueError): V.receipt(root, 'F0', 'same_lowmem')

    def test_pairing_and_exact_prestate_required(self):
        a = {k: 'x' for k in V.P.PAIR_KEYS}; a.update(prestate_sha256='a', frozen_exact=True)
        b = dict(a, prestate_sha256='b')
        V.paired_rows(a, b, False)
        with self.assertRaises(ValueError): V.paired_rows(a, b, True)
        b = dict(a, video='other')
        with self.assertRaises(ValueError): V.paired_rows(a, b, False)

    def test_probe_workflow_restores_canonical_not_own_previous_step(self):
        def make(weights, arm, lowmem):
            m, opt = toy()
            return m, opt, None, None, None, [p for p in m.parameters() if p.requires_grad]
        def labels(root, previous, old, weights, s, seed, step, reference):
            row = {k: ('paired' if k != 'step' else step) for k in V.P.PAIR_KEYS}
            row['domain'] = 'trackrad' if step%2 == 0 else 'lld_synthetic'
            return {}, row
        def execute(m, s, label, step, arm, params, classifier, opt, disc, head, tap, phases, audit, evidence):
            update(m, opt)
            return {}, dict(outputs=[torch.ones(2)],
                gradients={n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
                student=S.clone(dict(m.state_dict())), classifier={}, classifier_gradients={})
        class Phases:
            def __init__(self, callback): pass
            def report(self): return dict(phase_seconds={}, phase_memory={}, measured_seconds=.1)
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d, ExitStack() as stack:
            root = Path(d); previous = root/'old'; out = root/'output'
            V.C.save(previous/'manifest.json', dict(weights={}, files={'train-40': {}}))
            def mf(root, *args): V.C.freeze(root/'manifest.json', {'fixture': 1})
            replacements = [(V, 'manifest', mf), (V, 'make_model', make), (V, 'environment', lambda: {'cpu_fixture': True}),
                (V, 'execute', execute), (V, 'Phases', Phases),
                (V, 'memory', lambda: dict(peak_allocated=1000, device_total=100000)),
                (V, 'sample', lambda study, lld, data, seed, step: dict(video=torch.zeros(2, 3, 2, 2),
                    queries=torch.zeros(1, 2, 3), draw=100+step)),
                (V.P, 'checked', lambda *args: {}),
                (V.P, 'old_training', lambda *args: (None, [{}]*6, {'initial_sidecar': {}})),
                (V.B.BASE, 'input_labels', labels),
                (V.shutil, 'disk_usage', lambda *args: type('Disk', (), {'free': 30*2**30})())]
            for obj, name, value in replacements: stack.enter_context(patch.object(obj, name, value))
            for name in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
                stack.enter_context(patch.object(torch.cuda, name, lambda: None))
            for job in V.JOBS: V.probe(out, root, previous, root, root, root, root, 'F0', job)
            # All jobs complete and repeat uses the same hashes at every independent anchor.
            for job in V.JOBS: self.assertTrue(V.receipt(out, 'F0', job))
            original = V.C.read(out/'F0/original/steps.json')
            for job in ('same_repeat', 'same_lowmem'):
                for r in V.C.read(out/'F0'/job/'steps.json'):
                    self.assertEqual(r['prestate_sha256'], original[r['step']]['prestate_sha256'])
                    self.assertEqual(r['student_state_sha256'], original[r['step']]['student_state_sha256'])
            with patch.object(V, 'execute', side_effect=AssertionError('cache was not reused')):
                V.probe(out, root, previous, root, root, root, root, 'F0', 'lowmem')

    def test_restored_real_tracker_update_and_recompute(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, configure_sidecar_runtime
        torch.manual_seed(34)
        base = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        attach_sidecar(base, 'mamba'); configure_sidecar_runtime(base, backend='compact')
        V.configure(base, 'F010')
        with torch.no_grad(): base.updateformer.feature_sidecar.output.weight.uniform_(-.001, .001)
        torch.manual_seed(52)
        s = dict(video=torch.rand(3, 3, 64, 64)*255,
                 queries=torch.tensor([[[0., 20., 20.], [0., 30., 30.], [0., 40., 32.]]]),
                 domain='trackrad', draw=123)
        target = dict(tracks=s['queries'][:, None, :, 1:].expand(1, 3, 3, 2)+1,
                      visibility=torch.ones(1, 3, 3))
        def setup(m):
            params = [p for p in m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=5e-6)
            torch.manual_seed(86); head = V.DomainHead(m.latent_dim)
            disc = torch.optim.AdamW(head.parameters(), lr=1e-4)
            return params, opt, head, disc, V.Tap(m, 'fnet')
        class Phases:
            def part(self, name): return nullcontext()
        params, opt, head, disc, tap = setup(base)
        with patch.object(torch.Tensor, 'cuda', lambda t: t):
            V.execute(base, s, target, 0, 'F010', params, list(head.parameters()), opt, disc, head, tap,
                      Phases(), True, evidence=True)
        tap.close(); saved = S.capture(base, opt, head, disc)
        self.assertTrue(saved['optimizer']['state'], 'must test nonempty AdamW moments')
        results = []
        for lowmem in (False, False, True):
            m = copy.deepcopy(base)
            if lowmem: V.install(m)
            params, opt, head, disc, tap = setup(m)
            S.restore(saved, m, opt, head, disc)
            with patch.object(torch.Tensor, 'cuda', lambda t: t):
                _, e = V.execute(m, s, target, 1, 'F010', params, list(head.parameters()), opt, disc, head, tap,
                                 Phases(), True, evidence=True)
            tap.close()
            results.append(e)
        for e in results[1:]:
            groups, _ = V.A.compare_all(results[0], e, V.B.LIMITS, 1)
            self.assertTrue(all(g['within_existing_tolerance'] for g in groups.values()))


if __name__ == '__main__': unittest.main()
