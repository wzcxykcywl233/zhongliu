import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

import smoke as s
from acdc_data import GROUPS

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT / 'cotracker-algorithm/ext/co-tracker'))
torch.set_num_threads(1)


def fixture():
    rows = [{'patient': f'patient{i+1:03}', 'group': GROUPS[i//20], 'shape_tzyx': [30, 3, 32, 32]}
            for i in range(100)]
    assignment = s.split_patients(rows)
    for row in rows:
        row['split'] = assignment[row['patient']]
    shortest = next(r for r in rows if r['split'] == 'external-train-80')
    shortest['shape_tzyx'][0] = 12
    return {'audit_version': 2, 'training_ready': True, 'clip': s.clip_policy(),
            'split_counts': {'external-train-80': 80, 'external-dev-20': 20}, 'patients': rows}


class SmokeTests(unittest.TestCase):
    def test_jobs_cover_twelve_and_sixteen_all_teachers(self):
        manifest = fixture()
        jobs = s.make_jobs(manifest)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum(j['train_step'] for j in jobs), 5)
        self.assertEqual({(j['frames'], j['slot']) for j in jobs},
                         {(t, i) for t in (12, 16) for i in range(3)})
        self.assertEqual(jobs, s.make_jobs(manifest))
        train = {r['patient'] for r in manifest['patients'] if r['split'] == 'external-train-80'}
        self.assertTrue(all(j['patient'] in train for j in jobs))

    def test_missing_or_modified_audit_rejected(self):
        for field, value in [('audit_version', 1), ('training_ready', False), ('clip', {}),
                             ('split_counts', {'external-train-80': 100})]:
            manifest = fixture()
            manifest[field] = value
            with self.assertRaises(ValueError): s.make_jobs(manifest)
        manifest = fixture()
        manifest['patients'][0]['split'] = 'test-38'
        with self.assertRaises(ValueError): s.make_jobs(manifest)

    def test_no_shortest_dev_leak(self):
        manifest = fixture()
        next(r for r in manifest['patients'] if r['split'] == 'external-dev-20')['shape_tzyx'][0] = 10
        with self.assertRaisesRegex(ValueError, 'dev-only'): s.make_jobs(manifest)

    def test_safe_file(self):
        root = Path(os.environ.get('COTRACKER_TEST_TMP', tempfile.gettempdir())).resolve()
        for name in ('../escape', '/etc/test', 'C:/escape', 'a\\..\\escape'):
            with self.assertRaises(ValueError): s.safe_file(root, name)
        self.assertEqual(s.safe_file(root, 'attempts/file.pth'), root / 'attempts/file.pth')

    def test_frozen_inputs_and_cache_corruption(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root = Path(tmp)
            s.frozen(root/'frozen-smoke.json', {'a': 1})
            s.frozen(root/'frozen-smoke.json', {'a': 1})
            with self.assertRaises(ValueError): s.frozen(root/'frozen-smoke.json', {'a': 2})
            self.assertIsNone(s.load_verified(root, 'absent.json'))
            s.atomic_torch(root/'attempts/x.pth', {'x': torch.ones(1)})
            marker = {'complete': True, 'plan_sha256': s.digest(root/'frozen-smoke.json'),
                      'files_sha256': {'attempts/x.pth': s.digest(root/'attempts/x.pth')}}
            s.save(root/'A1-verified.json', marker)
            self.assertEqual(s.load_verified(root, 'A1-verified.json'), marker)
            s.atomic_torch(root/'attempts/x.pth', {'x': torch.zeros(1)})
            with self.assertRaises(ValueError): s.load_verified(root, 'A1-verified.json')

    def test_loss_matches_existing_coordinate_objective(self):
        from cotracker.models.core.cotracker.losses import sequence_loss
        gt = torch.randn(1, 12, 3, 2)
        vis = torch.tensor([.2, .95, 1.])[None, None].expand(1, 12, 3)
        predictions = [[gt + .1, gt + .2]]
        valid = torch.ones_like(vis)
        output = (None, None, None, (predictions, None, None, valid))
        actual = s.training_loss(output, {'tracks': gt, 'visibility': vis})
        expected = .05 * sequence_loss(predictions, [gt], [valid], vis=[(vis>.9).float()],
            gamma=.8, add_huber_loss=True, loss_only_for_visible=True)
        expected += .01 * sequence_loss(predictions, [gt], [valid], vis=[(vis<=.9).float()],
            gamma=.8, add_huber_loss=False, loss_only_for_visible=True)
        self.assertTrue(torch.equal(actual, expected))

    def test_loss_not_proportional_to_native_clip_length(self):
        scores = []
        for frames in (12, 16):
            gt = torch.zeros(1, frames, 3, 2)
            valid = torch.ones(1, frames, 3)
            prediction = gt + 1.
            scores.append(s.training_loss((None, None, None, ([[prediction]], None, None, valid)),
                                          {'tracks': gt, 'visibility': valid}))
        self.assertTrue(torch.allclose(scores[0], scores[1], atol=1e-6))

    def test_change_guard_includes_buffers(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer('fixed', torch.ones(1))
        before = s.state_hashes(model)
        with self.assertRaises(ValueError): s.check_changes(model, before, ['weight'])
        with torch.no_grad(): model.weight.add_(1)
        self.assertEqual(s.check_changes(model, before, ['weight']), ['weight'])
        model.fixed.add_(1)
        with self.assertRaises(ValueError): s.check_changes(model, before, ['weight'])

    def test_cpu_route_update_strict_reload_and_handoff(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        fresh_hashes = []
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            for route in s.ROUTES:
                with self.subTest(route=route):
                    torch.manual_seed(s.SEED)
                    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
                    torch.manual_seed(s.STUDENT_SEED)
                    report = s.configure(model, route)
                    before = s.state_hashes(model)
                    if route == 'A1': fresh_hashes.append({n: h for n, h in before.items() if n.startswith(s.PREFIX)})
                    video = torch.rand(1, 12, 3, 64, 64)*255
                    q = torch.tensor([[[0., 20., 20.], [11., 40., 40.]]])
                    result = model(video=video, queries=q, iters=2, is_train=True)
                    targets = {'tracks': result[0].detach() + 1., 'visibility': torch.ones(1, 12, 2)}
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                    s.training_loss(result, targets).backward()
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()))
                    self.assertFalse(any(p.grad is not None for p in model.parameters() if not p.requires_grad))
                    optimizer.step()
                    s.check_changes(model, before, report['trainable_names'])
                    path = Path(tmp)/(route+'.pth')
                    s.atomic_torch(path, {'model': model.state_dict(), 'route': route, 'steps': 5})
                    loaded = s.reload_checkpoint(path, route)
                    self.assertEqual(s.state_hashes(model), s.state_hashes(loaded))
                    model.eval(); loaded.eval()
                    with torch.no_grad():
                        one = model(video=video, queries=q, iters=2)[:3]
                        two = loaded(video=video, queries=q, iters=2)[:3]
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(one, two)))
                    handoff = s.configure_handoff(loaded, route)
                    if route != 'A1': fresh_hashes.append(handoff['sidecar_hashes'])
                    else:
                        self.assertEqual(handoff['sidecar_hashes'], {n: h for n, h in s.state_hashes(model).items() if n.startswith(s.PREFIX)})
                    self.assertTrue(all(p.requires_grad == n.startswith(s.PREFIX) for n, p in loaded.named_parameters()))
                    p, v, c = loaded(video=video, queries=q, iters=2, is_train=True)[:3]
                    (p.square().mean()*.001+v.square().mean()+c.square().mean()).backward()
                    self.assertFalse(any(p.grad is not None for n, p in loaded.named_parameters() if not n.startswith(s.PREFIX)))
                    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for n, p in loaded.named_parameters() if n.startswith(s.PREFIX)))
                    del model, loaded, optimizer, result, p, v, c
            self.assertTrue(all(h == fresh_hashes[0] for h in fresh_hashes))

    def test_teacher_protocol_baseline_unchanged(self):
        p = s.load_protocol(CHECKOUT)
        self.assertEqual([x['support'] for x in p.baseline()], [5, 0, 100])
        self.assertEqual([x['iterations'] for x in p.baseline()], [4, 4, 4])
        self.assertTrue(all(x['transform'] == x['intensity'] == 'none' for x in p.baseline()))

    def test_random_weight_teacher_interfaces_return_native_lengths(self):
        # Code-only CPU coverage; the remote smoke still must use actual weights and MRI.
        from cotracker.models.build_cotracker import build_cotracker
        from cotracker.models.core.model_utils import get_points_on_a_grid, get_uniformly_sampled_pts
        protocol = s.load_protocol(CHECKOUT)
        for slot in range(3):
            model = build_cotracker(None, offline=slot == 2, v2=slot == 0,
                                    window_len=60 if slot == 2 else 16).eval().requires_grad_(False)
            before = s.state_hashes(model)
            for frames in (12, 16):
                video = torch.rand(1, frames, 3, 64, 64)*255
                queries = torch.tensor([[[0., 16., 16.], [frames-1., 40., 40.]]])
                if slot < 2:
                    xy = get_points_on_a_grid(5, [64, 64], device='cpu')
                    support = torch.cat((torch.zeros_like(xy[..., :1]), xy), -1)
                else:
                    support = get_uniformly_sampled_pts(100, frames, [64, 64], device='cpu')
                p, v, c, audit = protocol.teacher_forward(slot, s.TEACHER_TYPES[slot], model, video,
                    queries, support, 'control', protocol.baseline())
                self.assertEqual(p.shape, (1, frames, 2, 2))
                self.assertEqual(v.shape, (1, frames, 2))
                self.assertEqual(c.shape, v.shape)
                self.assertTrue(all(bool(torch.isfinite(x).all()) for x in (p, v, c)))
                self.assertEqual(audit['support_used'], (25, 0, 100)[slot])
            self.assertEqual(before, s.state_hashes(model))


if __name__ == '__main__':
    unittest.main()
