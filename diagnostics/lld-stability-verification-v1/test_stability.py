import copy
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import support as H
import study as D

V, C, S = H.V, H.C, H.S
torch.set_num_threads(2)


def row(step):
    r = {k: 'paired-'+str(step) for k in H.P.PAIR_KEYS}
    r.update(step=step, domain='trackrad' if step%2 == 0 else 'lld_synthetic',
             point=.1, domain_loss=.2, student_grad_norm=.1, classifier_grad_norm=.2,
             grl_strength=0., seconds=2., peak_allocated=1000, device_total=100000)
    return r


def probe_row(n, i):
    return dict(row(i), after_updates=n, sample_index=i, point_loss=.1,
                coordinate_shift_rms_px=.01*n, visibility_mean_abs_change=0.,
                confidence_mean_abs_change=0., visibility_mean=.9, confidence_mean=.9,
                out_of_bounds_fraction=0.)


def toy():
    torch.manual_seed(35)
    m = torch.nn.Module()
    m.fnet = torch.nn.Module(); m.fnet.layer4 = torch.nn.Linear(2, 2)
    m.updateformer = torch.nn.Module(); m.updateformer.feature_sidecar = torch.nn.Linear(2, 2)
    m.register_buffer('frozen_buffer', torch.ones(1))
    m.eval(); m.updateformer.feature_sidecar.train()
    return m


class GraphTests(unittest.TestCase):
    def test_actual_domain_loss_detaches_zero_not_just_zero_gradient(self):
        m = toy(); h = V.DomainHead(2)
        e = m.fnet.layer4(torch.tensor([[1., 2.]]))
        task = m.updateformer.feature_sidecar(e).square().sum()
        for domain in ('trackrad', 'lld_synthetic'):
            proof = H.graph_check(m, h, e, task, domain)
            self.assertTrue(proof['zero_embedding_gradient_is_none'])
            self.assertGreater(proof['positive_grl_encoder_grad_norm'], 0)
        self.assertTrue(all(p.grad is None for p in list(m.parameters())+list(h.parameters())))
        import domain
        with patch.object(domain, 'reversal', lambda x, s: domain.Reverse.apply(x, s)):
            with self.assertRaisesRegex(ValueError, 'graph path'): H.graph_check(m, h, e, task, 'trackrad')
        with patch.object(domain, 'reversal', lambda x, s: x.detach()):
            with self.assertRaisesRegex(ValueError, 'gradients missing'): H.graph_check(m, h, e, task, 'trackrad')

    def test_real_tracker_lowmem_graph_and_fixed_probe_preserve_state(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, configure_sidecar_runtime
        torch.manual_seed(34)
        m = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        attach_sidecar(m, 'mamba'); configure_sidecar_runtime(m, backend='compact')
        V.configure(m, 'Fprobe'); V.install(m)
        h = V.DomainHead(m.latent_dim); tap = V.Tap(m, 'fnet'); tap.reset(3)
        s = dict(video=torch.rand(3, 3, 64, 64)*255,
            queries=torch.tensor([[[0., 20., 20.], [0., 30., 30.], [0., 40., 32.]]]),
            domain='trackrad', draw=123)
        label = dict(tracks=s['queries'][:, None, :, 1:].expand(1, 3, 3, 2)+1,
                     visibility=torch.ones(1, 3, 3))
        before = H.T.S.state_hashes(m)
        out = m(video=s['video'][None], queries=s['queries'], iters=4, is_train=True)
        task = H.T.S.training_loss(out, label)
        proof = H.graph_check(m, h, tap.embedding(), task, s['domain'])
        self.assertEqual(proof['zero_student_gradient_paths'], 0)
        self.assertEqual(len(tap.values), 1, 'recompute cannot duplicate hook')
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        del out, task; tap.values.clear()
        rng = S.fingerprint(S.rng_state())
        with patch.object(torch.Tensor, 'cuda', lambda x: x):
            first, packet = D.fixed_probe(m, tap, s, label)
            second, _ = D.fixed_probe(m, tap, s, label, packet)
        tap.close()
        self.assertEqual(first, second)
        self.assertEqual(second['coordinate_shift_rms_px'], 0)
        self.assertEqual(before, H.T.S.state_hashes(m))
        self.assertEqual(rng, S.fingerprint(S.rng_state()))


class GuardTests(unittest.TestCase):
    def test_runtime_memory_and_nonfinite_fail_closed(self):
        H.validate_records([row(0), row(1)])
        for field, value in [('seconds', 180.01), ('seconds', float('nan')), ('point', float('inf')),
                             ('student_grad_norm', -1), ('peak_allocated', 85001), ('device_total', float('inf'))]:
            with self.subTest(field=field, value=value):
                r = row(0); r[field] = value
                with self.assertRaises(ValueError): H.validate_records([r])
        with self.assertRaises(ValueError): H.validate_records([row(1)])

    def test_fixed_probes_complete_and_finite(self):
        rows = [probe_row(n, i) for n in (0, 25) for i in H.RULE['fixed_probe_sample_indices']]
        H.validate_probes(rows, 30)
        with self.assertRaises(ValueError): H.validate_probes(rows[:-1], 30)
        rows[-1]['visibility_mean'] = 2
        with self.assertRaises(ValueError): H.validate_probes(rows, 30)

    def test_loss_review_not_quality_gate(self):
        self.assertFalse(H.loss_alert(.1, .2))
        self.assertFalse(H.loss_alert(.001, .005))
        self.assertTrue(H.loss_alert(.1, .21))
        with self.assertRaises(ValueError): H.loss_alert(.1, float('nan'))
        self.assertFalse(H.RULE['full_training_authorized'])
        self.assertFalse(H.RULE['previous_strict_gate_changed'])
        self.assertEqual(H.RULE['updates'], 100)

    def test_receipt_files_must_be_complete_and_untampered(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root = Path(d); C.save(root/'plan.json', {}); C.save(root/'environment.json', {})
            C.save(root/'graph/summary.json', {'complete': True})
            H.publish(root, root/'graph', ['summary.json'], 'graph')
            self.assertTrue(H.completed(root, root/'graph', 'graph'))
            C.save(root/'graph/summary.json', {'complete': False})
            with self.assertRaises(ValueError): H.completed(root, root/'graph', 'graph')
            H.publish(root, root/'graph', [], 'graph')
            with self.assertRaises(ValueError): H.completed(root, root/'graph', 'graph')

    def test_old_evidence_binding_verifies_20_jobs_without_rerunning(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d:
            root = Path(d)
            with patch.object(H.B, 'identity', return_value={'upstream': 1}), patch.object(V, 'sources', return_value={}):
                C.save(root/'manifest.json', dict(rule=V.RULE, upstream={'upstream': 1}, sources={}))
                C.save(root/'environment.json', {'cpu': True})
                receipts = {}
                for arm in H.ARMS:
                    for job in V.JOBS:
                        f = V.folder(root, arm, job); C.save(f/'steps.json', [])
                        C.save(f/'complete.json', dict(complete=True, arm=arm, job=job,
                            manifest=C.sha(root/'manifest.json'), environment_sha256=C.sha(root/'environment.json'),
                            files={'steps.json': C.sha(f/'steps.json')}))
                        receipts[arm+'/'+job] = C.sha(f/'complete.json')
                C.save(root/'reports/all/summary.json', dict(complete=True, arms=list(H.ARMS), gate_changed=False,
                    manifest_sha256=C.sha(root/'manifest.json'), job_receipts=receipts))
                with patch.object(V, 'receipt', return_value=True) as deep:
                    binding = H.evidence_binding(root, root, root, root, deep=True)
                    self.assertEqual(deep.call_count, 20)
                    self.assertEqual(len(binding), 43)
                C.save(root/'F0/original/steps.json', [1])
                with self.assertRaises(ValueError): H.evidence_binding(root, root, root, root)


class WorkflowTests(unittest.TestCase):
    def test_100_updates_resume_pair_report_and_failure_not_skipped(self):
        class Phases:
            def part(self, name): return nullcontext()
            def report(self): return dict(phase_seconds={}, phase_memory={}, measured_seconds=.01)
        def make(weights, arm, lowmem):
            self.assertTrue(lowmem)
            m = toy(); params = list(m.parameters()); opt = torch.optim.AdamW(params, lr=5e-5)
            return m, opt, None, None, None, params
        def sample(root, previous, weights, lld, data, state, step):
            r = {k: row(step)[k] for k in H.P.PAIR_KEYS}
            C.freeze(root/f'inputs/seed_0/{step:05d}.json', r)
            return dict(video=torch.zeros(3, 3, 2, 2), queries=torch.zeros(1, 2, 3)), {}, r
        calls = []
        def execute(m, s, label, step, arm, params, classifier, opt, disc, head, tap, phases, audit, evidence):
            calls.append((arm, step)); opt.zero_grad(set_to_none=True)
            m.updateformer.feature_sidecar(m.fnet.layer4(torch.ones(1, 2))).square().sum().backward(); opt.step()
            return {k: row(step)[k] for k in ('point', 'domain_loss', 'student_grad_norm', 'classifier_grad_norm', 'grl_strength')}, None
        def probes(root, out, previous, weights, lld, data, state, model, tap, count, packets, records):
            for i in H.RULE['fixed_probe_sample_indices']:
                sample(root, previous, weights, lld, data, state, i)
                if count == 0: packets[i] = [torch.ones(1)]
                records.append(probe_row(count, i))
            C.save(out/'probes.json', records)
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as d, ExitStack() as stack:
            root = Path(d)
            def plan(root, *args, **kwargs): C.freeze(root/'plan.json', {'fixture': 1})
            def env(root, evidence): C.freeze(root/'environment.json', {'cpu': True})
            plan(root); env(root, root)
            C.save(root/'graph/summary.json', {'complete': True})
            H.publish(root, root/'graph', ['summary.json'], 'graph')
            init = H.T.S.state_hashes(toy()); proof = {'initial_sidecar': {n: h for n, h in init.items() if n.startswith(H.P.PREFIX)}}
            replacements = [(H, 'plan', plan), (H, 'check_environment', env),
                (D, 'inputs', lambda *a: ({}, {}, [], proof)), (D, 'get_sample', sample),
                (D, 'check_probes', probes), (V, 'make_model', make), (V, 'Phases', Phases),
                (V, 'execute', execute), (V, 'memory', lambda: dict(peak_allocated=1000, device_total=100000))]
            for obj, name, val in replacements: stack.enter_context(patch.object(obj, name, val))
            for name in ('reset_peak_memory_stats', 'synchronize'): stack.enter_context(patch.object(torch.cuda, name, lambda: None))
            def interrupted(*args, **kwargs):
                if args[3] == 13: raise RuntimeError('simulated power interruption')
                return execute(*args, **kwargs)
            with patch.object(V, 'execute', interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interruption'): D.train(*([root]*8), 'F0')
            saved = H.load(root/'train/F0/resume.pth')
            self.assertEqual(len(saved['rows']), 10)
            calls.clear()
            for arm in H.ARMS: D.train(*([root]*8), arm)
            self.assertEqual(calls[0], ('F0', 10))
            for arm in H.ARMS:
                rows = C.read(root/'train'/arm/'steps.json')
                self.assertEqual(len(rows), 100)
                self.assertEqual(rows[0]['queries'], 'paired-0', 'do not overwrite query fingerprint with count')
                self.assertEqual(len(C.read(root/'train'/arm/'probes.json')), 20)
            with patch.object(V, 'execute', side_effect=AssertionError('cached job reran')):
                D.train(*([root]*8), 'F0')
            D.report(*([root]*5))
            summary = C.read(root/'summary.json')
            self.assertTrue(summary['paired_inputs_and_labels'])
            self.assertFalse(summary['full_training_authorized'])
            self.assertFalse(summary['numerical_equivalence_claimed'])
            self.assertFalse(summary['previous_strict_gate_changed'])
            f = root/'train/F010'; p = C.read(f/'probes.json'); p[-1]['point_loss'] = .3
            C.save(f/'probes.json', p); H.publish(root, f, ['resume.pth', 'steps.json', 'probes.json', 'proof.json'], 'F010')
            D.report(*([root]*5))
            self.assertTrue(C.read(root/'summary.json')['arms']['F010']['needs_loss_review'])
            p = C.read(f/'proof.json'); p['frozen_exact'] = False; C.save(f/'proof.json', p)
            H.publish(root, f, ['resume.pth', 'steps.json', 'probes.json', 'proof.json'], 'F010')
            with self.assertRaisesRegex(ValueError, 'training proof'): D.report(*([root]*5))
            p['frozen_exact'] = True; C.save(f/'proof.json', p)
            p = C.read(f/'steps.json'); p[0]['queries'] = 'different'; C.save(f/'steps.json', p)
            H.publish(root, f, ['resume.pth', 'steps.json', 'probes.json', 'proof.json'], 'F010')
            with self.assertRaisesRegex(ValueError, 'not paired'): D.report(*([root]*5))
            # A prior failed arm has no completion marker; rerun must stop before training.
            with patch.object(H, 'completed', side_effect=lambda r, f, k: k == 'graph'):
                C.save(root/'train/F0/failure.json', {'review_required': True})
                with self.assertRaisesRegex(ValueError, 'requires review'): D.train(*([root]*8), 'F0')


if __name__ == '__main__': unittest.main()
