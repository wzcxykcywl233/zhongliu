"""CPU regression tests; real GPU smoke is mandatory before automatic training."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
os.environ['COTRACKER_APP_DIR'] = str(REPO / 'cotracker-algorithm')
os.environ['SIDECAR_CHECKOUT'] = str(REPO)
sys.path.insert(0, str(PACKAGE))
import protocol as P
import study as S
import train
import torch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def metric(dsc=.9, d98=.9, hd95=4.):
    return dict(DSC=dsc, D98=d98, HD95=hd95, MASD=1.5, CD=1.7, TimeSec=200.)


class FakeTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eval()
        self.seen = None

    def forward(self, video, queries, iters=4):
        self.seen = queries.clone(), iters
        shape = (1, video.shape[1], queries.shape[1])
        p = queries[..., 1:].unsqueeze(1).repeat(1, shape[1], 1, 1) + .1 * iters + queries.shape[1] / 1000
        return p, torch.full(shape, .95), torch.full(shape, .8), None


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root = Path(self.tmp.name)
        self.pools, self.groups = P.matrix()

    def tearDown(self):
        self.tmp.cleanup()

    def test_baseline_is_retained_if_candidates_degrade(self):
        scores = {'control': [metric()], 'a': [metric(.899)], 'b': [metric(.901, .895)]}
        self.assertEqual(P.best(scores, ['a', 'b']), 'control')
        scores['b'] = [metric(.901, .899)]
        self.assertEqual(P.best(scores, ['a', 'b']), 'b')
        scores['b'] = [metric(.901, .9, 4.1)]
        self.assertEqual(P.best(scores, ['a', 'b']), 'control')

    def test_full_selection_requires_two_improved_seeds(self):
        reference = [metric()] * 3
        self.assertFalse(P.admissible([metric(.904), metric(.8999), metric(.8999)], reference))
        self.assertTrue(P.admissible([metric(.901), metric(.901), metric(.8999)], reference))
        self.assertFalse(P.admissible([metric(.91), metric(.91), metric(.897)], reference))
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            P.admissible([metric()], reference)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            P.admissible([metric(float('nan'))], [metric()])

    def test_all_transforms_preserve_time_rng_dtype_and_student_input(self):
        video = torch.arange(1 * 60 * 3 * 32 * 48, dtype=torch.float32).reshape(1, 60, 3, 32, 48).remainder(251)
        queries = torch.tensor([[[0., 3., 3.], [29., 22., 23.], [59., 42., 28.]]])
        for family, names in self.groups.items():
            slot = P.FAMILY_SLOT[family]
            draw = torch.zeros(1, 25 if slot < 2 else 100, 3)
            draw[..., 0], draw[..., 1], draw[..., 2] = 17, 24, 20
            for name in names:
                model = FakeTeacher()
                rng, vcopy, qcopy, dcopy = torch.get_rng_state().clone(), video.clone(), queries.clone(), draw.clone()
                p, v, c, audit = P.teacher_forward(slot, ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot],
                                                   model, video, queries, draw, name, self.pools[name])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(torch.equal(video, vcopy) and torch.equal(queries, qcopy) and torch.equal(draw, dcopy))
                self.assertTrue(torch.equal(model.seen[0][:, :3, 0], queries[..., 0]))
                expected_count = self.pools[name][slot]['support'] ** 2 if slot < 2 else self.pools[name][slot]['support']
                self.assertEqual(model.seen[0].shape[1], 3 + expected_count)
                mode = self.pools[name][slot]['transform']
                self.assertEqual(audit['teacher_calls'], len(P.view_modes(mode)))
                self.assertEqual(audit['visibility_sha256'], audit['base_visibility_sha256'])
                self.assertEqual(audit['confidence_sha256'], audit['base_confidence_sha256'])
                self.assertEqual(audit['transform']['mode'], self.pools[name][slot]['transform'])
                self.assertEqual(audit['pool_sha256'], P.digest(self.pools[name]))
                self.assertEqual(p.shape, (1, 60, 3, 2))
                offset = .1 * self.pools[name][slot]['iterations'] + (3 + expected_count) / 1000
                expected = queries[..., 1:].unsqueeze(1).repeat(1, 60, 1, 1) + offset
                for t in range(60):
                    for j, q in enumerate(queries[0]):
                        if t == q[0] or (slot < 2 and t < q[0]) or mode == 'none':
                            continue
                        if mode in ('xmean', 'xguard'):
                            expected[0, t, j, 0] -= offset
                        else:
                            expected[0, t, j] -= 2 * offset / 3
                self.assertTrue(torch.allclose(p, expected, atol=1e-5))
                if slot == 0:
                    self.assertTrue(bool((c == 1).all()))

    def test_flip_mapping_is_involution_including_boundaries(self):
        video = torch.arange(24, dtype=torch.float32).reshape(1, 2, 1, 3, 4)
        q = torch.tensor([[[0., 0., 0.], [1., 3., 2.], [1., 1.25, 1.5]]])
        support = q[:, :1]
        for mode in ('flip_x', 'flip_y'):
            v, qq, ss, _ = P.transform_inputs(video, q, support, mode)
            v2, q2, s2, _ = P.transform_inputs(v, qq, ss, mode)
            self.assertTrue(torch.equal(v2, video) and torch.equal(q2, q) and torch.equal(s2, support))
            output = P.inverse_tracks(qq[..., 1:].unsqueeze(1), mode, 3, 4)
            self.assertTrue(torch.equal(output[:, 0], q[..., 1:]))

    def test_fusion_gates_boundaries_anchors_and_invalid_views(self):
        q = torch.tensor([[[1., 4., 4.]]])
        p = torch.full((1, 3, 1, 2), 4.)
        v, c = torch.full((1, 3, 1), .95), torch.full((1, 3, 1), .8)
        base = (p, v, c)
        other = (p + torch.tensor([2., 0.]), v.clone(), c.clone())
        out = P.fuse_views([base, other], q, 'xguard', 16, 16, True)
        self.assertTrue(torch.equal(out[0][0, :, 0, 0], torch.tensor([5., 4., 5.])))
        self.assertIs(out[1], v); self.assertIs(out[2], c)
        self.assertEqual(out[3]['accepted_point_frames'], 2)
        online = P.fuse_views([base, other], q, 'xguard', 16, 16, False)
        self.assertTrue(torch.equal(online[0][0, :, 0, 0], torch.tensor([4., 4., 5.])))
        for bad in ((p + torch.tensor([2.001, 0.]), v, c),
                    (p + 1, torch.full_like(v, .9), c), (p + 1, v, torch.full_like(c, .49)),
                    (p + 20, v, c), (p * float('nan'), v, c)):
            got = P.fuse_views([base, bad], q, 'xguard', 16, 16, True)
            self.assertTrue(torch.equal(got[0], p))
            self.assertEqual(got[3]['fallback_point_frames'], 2)
        left, right = (p - torch.tensor([1.5, 0.]), v, c), (p + torch.tensor([1.5, 0.]), v, c)
        self.assertTrue(torch.equal(P.fuse_views([base, left, right], q, 'xyguard', 16, 16, True)[0], p))
        mean = P.fuse_views([base, (p + 4, v * 0, c * 0)], q, 'xmean', 16, 16, True)[0]
        self.assertTrue(torch.equal(mean[0, :, 0, 0], torch.tensor([6., 4., 6.])))
        zero = P.fuse_views([base, base], q, 'xguard', 16, 16, True)
        self.assertTrue(torch.equal(zero[0], p)); self.assertEqual(zero[3]['changed_point_frames'], 0)
        fractional = (p + .1234567, v, c)
        zero3 = P.fuse_views([fractional] * 3, q, 'xyguard', 16, 16, True)
        self.assertTrue(torch.equal(zero3[0], fractional[0]))
        self.assertEqual(zero3[3]['changed_point_frames'], 0)
        with self.assertRaisesRegex(ValueError, 'Invalid original'):
            P.fuse_views([(p.double(), v, c)], q, 'none', 16, 16, True)
        with self.assertRaisesRegex(ValueError, 'view count'):
            P.fuse_views([base], q, 'xguard', 16, 16, True)
        video = torch.zeros(1, 3, 3, 16, 16)
        for bad in (video.double(), video - 500, video * float('nan')):
            with self.assertRaises(ValueError):
                P.transform_inputs(bad, q, q[:, :0], 'flip_x')
        with self.assertRaises(ValueError):
            P.transform_inputs(video, q, q[:, :0], 'gamma')

    def test_selection_diagnostics_explain_seed_failure(self):
        reference = [metric()] * 3
        deltas = [-.0003813544619362119, .0006662706676086039, -.00010789705301961217]
        evidence = P.selection_evidence([metric(.9 + d) for d in deltas], reference)
        self.assertFalse(evidence['Eligible'])
        self.assertEqual(evidence['FailedRules'], 'improved_seeds')
        self.assertEqual(evidence['ImprovedSeeds'], 1)
        self.assertGreater(evidence['Delta_DSC'], .00001)

    def test_baseline_equals_previous_selected_pool(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('prior_search_protocol', REPO / 'diagnostics/teacher-pool-search-v1/protocol.py')
        prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
        expected = prior.matrix()[0]['on3_s0']
        actual = [{k: v for k, v in s.items() if k not in ('transform', 'intensity')} for s in P.baseline()]
        self.assertEqual(actual, expected)

    def test_noop_matches_original_and_wrong_teacher_modes_fail(self):
        video = torch.zeros(1, 60, 3, 64, 64)
        queries = torch.tensor([[[29., 22., 33.]]])
        draw = torch.zeros(1, 100, 3)
        model = FakeTeacher()
        expected = model(video, torch.cat((queries, draw), 1))
        actual = P.teacher_forward(2, 'offline_cotracker_three', model, video, queries, draw, 'control', self.pools['control'])
        for i in range(3):
            self.assertTrue(torch.equal(expected[i][:, :, :1], actual[i]))
        model.train()
        with self.assertRaisesRegex(ValueError, 'unfrozen'):
            P.teacher_forward(2, 'offline_cotracker_three', model, video, queries, draw, 'control', self.pools['control'])
        model.eval()
        queries[..., 0] = 60
        with self.assertRaisesRegex(ValueError, 'query time'):
            P.teacher_forward(2, 'offline_cotracker_three', model, video, queries, draw, 'control', self.pools['control'])

    def test_instrumentation_preserves_loss_and_freezing_guards(self):
        source = (S.APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8')
        altered = P.instrument(source, S.legacy_protocol.instrument)
        compile(altered, 'trainer-adapter', 'exec')
        for expression in ('assert_sidecar_training_mode(_unwrap_model(model))',
                           'self.backward(scaler.scale(loss))', 'torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)',
                           'vis_g = vis_g > 0.9', 'window_len=16', 'uniform_size = 100', 'grid_size = args.train_grid_size'):
            self.assertIn(expression, altered)
        self.assertEqual(altered.count('if total_steps >= __stop_steps__:'), 2)
        self.assertIn("final['teacher_pool_sha256'] = __pool_sha__", altered)
        with self.assertRaisesRegex(ValueError, 'anchors'):
            P.instrument(source.replace('    vis_g = vis_g > 0.9', '    vis_g = vis_g > 0.8'), S.legacy_protocol.instrument)

    def fake_rows(self, arm='control'):
        cfg = P.config(0, arm, 'smoke', self.pools[arm], 'fixture')
        sampler = S.L.TeacherPairSampler(3, seed=P.BASE_SEED)
        rows = []
        for i in range(10):
            slot = sampler.sample().primary
            setting = cfg['pool'][slot]
            code = 'b' * 64 if setting != self.pools['control'][slot] else 'a' * 64
            audit = {'policy': arm, 'pool_sha256': cfg['pool_sha256'], 'teacher_setting': setting,
                     'teacher_type': ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot],
                     'support_used': setting['support'] ** 2 if slot < 2 else setting['support'], 'iterations': setting['iterations'],
                     'transform': {'mode': setting['transform'], 'fallback': False}, 'teacher_calls': 1,
                     'output_coordinates': 'original', 'teacher_queries_sha256': 'a' * 64,
                     'teacher_video_sample_sha256': 'b' * 64,
                     'query_time_counts': [128] + [0] * 59, 'support_draw_sha256': 'c' * 64,
                     **{k: code for k in P.LABEL_KEYS}, 'teacher_seconds': .1, 'visible_gt09_fraction': .9,
                     'confidence_mean': .99, 'out_of_bounds_fraction': 0.}
            base_code = 'b' * 64 if any(setting[k] != self.pools['control'][slot][k] for k in ('support', 'iterations', 'intensity')) else 'a' * 64
            mode = setting['transform']
            audit.update(teacher_calls=len(P.view_modes(mode)), ensemble_rule=P.ENSEMBLE_RULE,
                         intensity={'mode': setting['intensity'], 'fallback': False})
            if setting['intensity'] == 'percentile':
                audit['intensity'].update(low=1., high=250., samples=1000, sample_stride=1)
            audit['views'] = [{'mode': m, 'queries_sha256': 'a' * 64, 'video_sample_sha256': 'b' * 64,
                               **{k: base_code if j == 0 else 'c' * 64 for k in P.LABEL_KEYS}}
                              for j, m in enumerate(P.view_modes(mode))]
            for key in P.LABEL_KEYS:
                audit['base_' + key] = base_code
            audit['visibility_sha256'] = audit['confidence_sha256'] = base_code
            if mode == 'none': audit['tracks_sha256'] = base_code
            audit['ensemble'] = {'point_frames': 7680, 'eligible_point_frames': 7552,
                                 'accepted_point_frames': 7552 if mode != 'none' else 0,
                                 'changed_point_frames': 7552 if mode != 'none' else 0,
                                 'fallback_point_frames': 0, 'mean_disagreement_pixels': 0.,
                                 'mean_correction_pixels': 0., 'max_correction_pixels': 0.}
            rows.append({'step': i, 'architecture': 'base', 'teacher_policy': arm, 'label_audit': audit,
                         'case_names': ['A_001'], 'queries_sha256': 'd' * 64, 'video_sha256': 'e' * 64,
                         'primary_index': slot, 'auxiliary_index': None, 'loss': .1, 'step_seconds': 2.,
                         'health': {'peak_allocated_mib': 12000., 'peak_reserved_mib': 13000., 'driver_free_mib': 10000.}})
        return cfg, rows

    def parse_rows(self, cfg, rows):
        path = self.root / 'paired.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return P.records(path, cfg)

    def test_paired_each_slot_and_replay_checks(self):
        cfg, rows = self.fake_rows()
        reference = self.parse_rows(cfg, rows)
        self.parse_rows(cfg, rows + [copy.deepcopy(rows[0])])
        altered = copy.deepcopy(rows[0]); altered['video_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'Resumed'):
            self.parse_rows(cfg, rows + [altered])
        for arm in ('v2_e', 'on3_e', 'offscaled_ie', 'offscaled_e'):
            candidate_cfg, candidate_rows = self.fake_rows(arm)
            candidate = self.parse_rows(candidate_cfg, candidate_rows)
            changed = P.paired(reference, candidate, cfg['pool'], candidate_cfg['pool'])
            self.assertEqual(sum(v > 0 for v in changed), 1)
            unchanged = next(r for r in candidate.values() if cfg['pool'][r['primary_index']] == candidate_cfg['pool'][r['primary_index']])
            unchanged['label_audit']['tracks_sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'Unmodified'):
                P.paired(reference, candidate, cfg['pool'], candidate_cfg['pool'])

    def test_invalid_metadata_and_memory_are_rejected(self):
        cfg, rows = self.fake_rows()
        for field, value in (('support_used', 999), ('iterations', 99), ('pool_sha256', 'wrong')):
            changed = copy.deepcopy(rows)
            changed[0]['label_audit'][field] = value
            with self.assertRaisesRegex(ValueError, 'settings'):
                self.parse_rows(cfg, changed)
        rows[0]['health']['driver_free_mib'] = 10
        with self.assertRaisesRegex(ValueError, 'memory'):
            self.parse_rows(cfg, rows)

    def test_stages_have_fresh_weights_but_same_scheduler(self):
        write(self.root / 'manifest.json', {})
        for stage, phase, pools, seeds in (('pilot', 'pilot', self.pools, [0]),
                                    ('confirm', 'full', {'control': self.pools['control'], 'v2_si': self.pools['v2_si']}, [0, 1, 2])):
            S.make_plan(self.root, stage, phase, pools, seeds)
            S.get_plan(self.root, stage)
        pilot = P.read(self.root / 'pilot/train/seed_0/control/run-config.json')
        full = P.read(self.root / 'full/train/seed_0/control/run-config.json')
        self.assertEqual((pilot['steps'], full['steps']), (250, 1000))
        self.assertEqual(pilot['scheduler_num_steps'], full['scheduler_num_steps'])
        before = P.sha(self.root / 'full/train/seed_0/control/run-config.json')
        S.make_plan(self.root, 'confirm', 'full', {'control': self.pools['control'], 'v2_si': self.pools['v2_si']}, [0, 1, 2])
        self.assertEqual(before, P.sha(self.root / 'full/train/seed_0/control/run-config.json'))
        changed = copy.deepcopy(self.pools['control']); changed[2]['iterations'] = 8
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            S.make_plan(self.root, 'confirm', 'full', {'control': changed}, [0, 1, 2])

    def test_prepare_freezes_all_inputs_without_changing_original_study(self):
        checkout, prior, weights, data, out = (self.root / p for p in ('checkout', 'prior', 'weights', 'datasets', 'out'))
        checkout.mkdir(); prior.mkdir(); weights.mkdir()
        trainer = checkout / 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
        trainer.parent.mkdir(parents=True)
        trainer.write_text('# immutable trainer')
        inventory = []
        for split, alias, prefix, count in (('train-40', 'train', 'A', 40), ('validation-10', 'validation', 'B', 10), ('test-38', 'test', 'C', 38)):
            for i in range(count):
                case = f'{prefix}_{i:03}'
                path = data / alias / case / 'frames.mha'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'fixture-not-medical-data')
                inventory.append({'Split': split, 'Case': case, 'File': f'C:\\data\\{alias}\\{case}\\frames.mha', 'SHA256': P.sha(path)})
        checkpoints = []
        for name in ('scaled_offline.pth', 'baseline_offline.pth', 'baseline_online.pth', 'cotracker2v1.pth'):
            torch.save({'weight': torch.ones(2)}, weights / name)
            checkpoints.append({'Name': name, 'SHA256': P.sha(weights / name)})
        write(prior / 'frozen-study.json', {'Source': [{'File': trainer.relative_to(checkout).as_posix(), 'SHA256': P.sha(trainer)}],
                                          'Weights': checkpoints, 'Dataset': inventory})
        images = {k: 'sha256:' + c * 64 for k, c in (('Training', 'a'), ('Inference', 'b'), ('Evaluation', 'c'))}
        write(prior / 'local-runtime-images.json', images)
        write(prior / 'reference-images.json', {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']})
        original = {p.name: P.sha(p) for p in prior.iterdir()}
        with patch.object(S, 'CHECKOUT', checkout), patch.object(S, 'APP', checkout / 'cotracker-algorithm'), patch('freeze_sidecar_reference.os.fsync'):
            S.prepare(out, prior, PACKAGE, weights, data)
            S.prepare(out, prior, PACKAGE, weights, data)
            self.assertEqual(len(list(out.glob('*/train/seed_*/*/run-config.json'))), 51)
            self.assertFalse((out / 'plans/test.json').exists())
            self.assertEqual(original, {p.name: P.sha(p) for p in prior.iterdir()})
            trainer.write_text('# changed trainer')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                S.prepare(out, prior, PACKAGE, weights, data)

    def test_real_model_audit_rejects_base_or_teacher_pool_change(self):
        torch.set_num_threads(1)
        model = S.L.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        info = S.L.attach_sidecar(model, 'mamba')
        info.update(execution_runtime=P.RUNTIME, initialization_outputs_exact=True,
                    initialization_reference='frozen-pretrained', initialization_max_abs_difference=[0., 0., 0.])
        with torch.no_grad(): model.updateformer.feature_sidecar.output.weight.fill_(.001)
        manifest = {'trainer_sha256': 'a' * 64, 'package_sha256': {'train.py': 'b' * 64, 'protocol.py': 'c' * 64}}
        write(self.root / 'manifest.json', manifest)
        write(self.root / 'reference-proof.json', {'base_tensor_sha256': info['base_tensor_sha256']})
        plan = S.make_plan(self.root, 'smoke', 'smoke', {'control': self.pools['control']}, [0])
        folder = self.root / 'smoke/train/seed_0/control'
        cfg, rows = self.fake_rows()
        write(folder / 'initialization-report.json', {'architecture': 'base', 'initialization_outputs_exact': True, 'sidecar': info})
        source = P.instrument((S.APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8'), S.legacy_protocol.instrument)
        write(folder / 'trainer-adapter.json', {'source_sha256': 'a' * 64, 'adapter_sha256': 'b' * 64, 'protocol_sha256': 'c' * 64,
              'executed_source_sha256': S.hashlib.sha256(source.encode()).hexdigest(), 'pool_sha256': cfg['pool_sha256'], 'stop_steps': 10, 'scheduler_steps': 1000})
        (folder / 'paired-steps.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        final = {'total_steps': 10, 'feature_sidecar': 'mamba', 'backbone_architecture': 'base',
                 'teacher_student_policy': 'control', 'teacher_pool_sha256': cfg['pool_sha256'], 'model': model.state_dict()}
        with patch.object(torch, 'load', return_value=final):
            actual, _ = S.verify_arm(self.root, plan, 0, 'control')
            self.assertEqual(len(actual), 10)
            final['teacher_pool_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'pool'):
                S.verify_arm(self.root, plan, 0, 'control')
            final['teacher_pool_sha256'] = cfg['pool_sha256']
            key = next(k for k in final['model'] if '.feature_sidecar.' not in k)
            final['model'][key] = final['model'][key] + .01
            with self.assertRaisesRegex(ValueError, 'Frozen base'):
                S.verify_arm(self.root, plan, 0, 'control')

    def test_official_reporting_and_checkpoint_binding(self):
        import csv
        cases = ['A_010', 'A_012']
        cfg = {'border_points': 1000}
        write(self.root / 'manifest.json', {'case_ids': {'validation-10': cases}, 'student_config': cfg})
        write(self.root / 'reference-images.json', {'algorithm': 'sha256:' + 'a' * 64})
        (self.root / 'pretrained-frozen.pth').write_bytes(b'official-fixture')
        plan = S.make_plan(self.root, 'pilot', 'pilot', {'control': self.pools['control'], 'v2_e': self.pools['v2_e']}, [0])
        for arm, kind, score, seed in (('pretrained', 'none', .9, -1), ('control', 'mamba', .901, 0), ('v2_e', 'mamba', .902, 0)):
            if kind != 'none':
                (self.root / f'pilot/train/seed_0/{arm}/cotracker_three_final.pth').write_bytes(arm.encode())
            S.bind_evaluation(self.root, 'pilot', 'validation-10', seed, arm)
            relative = 'pretrained' if arm == 'pretrained' else 'seed_0/' + arm
            folder = self.root / 'pilot/validation-10' / relative
            values = dict(zip(S.L.FIELDS.values(), (score, 4., 1.5, 1.7, .9)))
            write(folder / P.PROFILE / 'metrics.json', {'aggregates': {**values, 'total_time': 200.},
                   'results': [{'case_id': case, **values} for case in cases]})
            write(folder / 'frozen-images.json', P.read(self.root / 'reference-images.json'))
            for case in cases:
                job = folder / P.PROFILE / 'checkpoint/jobs' / case
                output = job / 'output/images/mri-linac-series-targets/output.mha'
                output.parent.mkdir(parents=True)
                output.write_bytes(b'output-fixture')
                (job / '.complete').write_text('done')
                write(job / 'diagnostics.json', {'profile': P.PROFILE, 'config': cfg,
                      'model': {'parameters': 25385700 if kind == 'none' else 28763025, 'feature_sidecar': kind,
                                'frozen_pretrained': True, 'sidecar_runtime': P.RUNTIME},
                      'prediction': {'output_file_sha256': P.sha(output)},
                      'mechanism': {'sidecar_calls': 2, 'sidecar_residual_rms_max': .01}})
        with patch.object(S, 'checked_training', return_value={'artifacts_sha256': {}}):
            S.report(self.root, 'pilot', 'validation-10')
            S.report(self.root, 'pilot', 'validation-10')
            with (self.root / 'reports/pilot/validation-10/summary.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(next(r for r in rows if r['Arm'] == 'v2_e')['DSC_Mean']), .902)
            S.checked_report(self.root, 'pilot')
            with self.assertRaisesRegex(ValueError, 'Only frozen'):
                S.report(self.root, 'pilot', 'test-38')
            checkpoint = self.root / 'pilot/train/seed_0/v2_e/cotracker_three_final.pth'
            checkpoint.write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                S.bind_evaluation(self.root, 'pilot', 'validation-10', 0, 'v2_e')
            with self.assertRaisesRegex(ValueError, 'checkpoint binding'):
                S.report(self.root, 'pilot', 'validation-10')

    def test_parent_artifact_changes_block_downstream_plan(self):
        write(self.root / 'manifest.json', {})
        artifact = self.root / 'validation-scores.json'
        write(artifact, {'DSC': .9})
        proof = self.root / 'reports/pilot/validation-10/audit.json'
        write(proof, {'artifacts_sha256': {'validation-scores.json': P.sha(artifact)}})
        parent = {proof.relative_to(self.root).as_posix(): P.sha(proof)}
        S.make_plan(self.root, 'confirm', 'full', {'control': self.pools['control']}, [0, 1, 2], parent)
        S.get_plan(self.root, 'confirm', deep=True)
        write(artifact, {'DSC': .99})
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            S.get_plan(self.root, 'confirm', deep=True)


    def test_factorial_matrix_joint_budget_and_no_standalone_gate(self):
        self.assertEqual(len(self.pools), 46)
        self.assertEqual(len({P.digest(p) for p in self.pools.values()}), 46)
        for slot, family in enumerate(P.FAMILIES):
            self.assertEqual(len(self.groups[family]), 16)
            counts = [0] * 5
            for a in self.groups[family]:
                changed = [s for s in range(3) if self.pools[a][s] != self.pools['control'][s]]
                self.assertEqual(changed, [] if a == 'control' else [slot])
                counts[len(P.changed_factors(self.pools[a][slot], self.pools['control'][slot]))] += 1
            self.assertEqual(counts, [1, 4, 6, 4, 1])
        scores = {a: [metric(.89)] for a in self.pools}
        scores['control'] = [metric(.90)]  # all single-slot changes lose, still explore!
        choices = P.local_candidates(scores, self.pools, self.groups)
        for slot in range(3):
            a, b = choices[str(slot)]
            self.assertNotEqual(a, b)
            self.assertGreaterEqual(len(P.changed_factors(self.pools[b][slot], self.pools['control'][slot])), 2)
        joint = P.joint_matrix(self.pools, choices)
        self.assertEqual([sum(P.slot_count(p, self.pools['control']) == n for p in joint.values()) for n in range(4)], [1, 6, 12, 8])
        all_pools = {**self.pools, **joint}
        self.assertEqual(len(all_pools), 66)
        scores.update({a: [metric(.89)] for a in joint if a not in scores})
        confirmed, picks = P.confirmation(scores, all_pools, joint)
        self.assertTrue(set(P.subset_closure(picks['best_three_pilot'], all_pools)) <= set(confirmed))
        self.assertGreaterEqual(len(confirmed), 8)
        self.assertLessEqual(len(confirmed), 12)
        bad = copy.deepcopy(choices); bad['0'] = choices['1']
        with self.assertRaisesRegex(ValueError, 'wrong teacher slot'):
            P.joint_matrix(self.pools, bad)

    def test_intensity_order_flat_fallback_and_support_prefix(self):
        video = torch.linspace(20, 180, 60*3*16*20).reshape(1,60,3,16,20)
        q = torch.tensor([[[0., 1., 2.], [59., 10., 11.]]])
        draw = torch.arange(300, dtype=torch.float32).reshape(1,100,3)
        draw[..., 0] = 0
        setting = self.pools['offscaled_sine'][2]
        normalized, support, meta = P.prepare_inputs(2, setting, video, q, draw)
        self.assertTrue(torch.equal(support, draw[:, :50]))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(float(normalized.min()), 0.)
        self.assertEqual(float(normalized.max()), 255.)
        self.assertFalse(meta['fallback'])
        lo, hi = torch.quantile(video.flatten(), torch.tensor([.01, .99]))
        self.assertTrue(torch.equal(normalized, ((video-lo)/(hi-lo)).clamp(0,1)*255))
        flipped, _, _, _ = P.transform_inputs(normalized, q, support, 'flip_x')
        self.assertTrue(torch.equal(flipped, normalized.flip(-1)))
        flat = torch.full_like(video, 80.)
        same, _, meta = P.prepare_inputs(2, setting, flat, q, draw)
        self.assertIs(same, flat); self.assertTrue(meta['fallback'])
        invalid = dict(setting, iterations=8)
        with self.assertRaisesRegex(ValueError, 'factorial'):
            P.prepare_inputs(2, invalid, video, q, draw)

    def test_three_slot_smoke_pair_audit_and_intensity_tamper(self):
        self.pools.update(P.smoke_pools(self.pools))
        cfg, rows = self.fake_rows()
        reference = self.parse_rows(cfg, rows)
        candidate_cfg, candidate_rows = self.fake_rows('j_sine_sine_sine')
        candidate = self.parse_rows(candidate_cfg, candidate_rows)
        changed = P.paired(reference, candidate, cfg['pool'], candidate_cfg['pool'])
        self.assertTrue(all(x > 0 for x in changed))
        bad = copy.deepcopy(candidate_rows)
        bad[0]['label_audit']['intensity']['high'] = 256.
        with self.assertRaisesRegex(ValueError, 'intensity'):
            self.parse_rows(candidate_cfg, bad)
        bad = copy.deepcopy(candidate_rows[0])
        bad['label_audit']['intensity']['low'] = 2.
        with self.assertRaisesRegex(ValueError, 'Resumed'):
            self.parse_rows(candidate_cfg, candidate_rows + [bad])

    def test_confirmation_budget_across_rankings(self):
        import random
        choices = {str(s): [f'{f}_e', f'{f}_sine'] for s, f in enumerate(P.FAMILIES)}
        joint = P.joint_matrix(self.pools, choices)
        all_pools = {**self.pools, **joint}
        rng = random.Random(31)
        for _ in range(50):
            scores = {a: [metric(.89 + .02*rng.random())] for a in all_pools}
            confirmed, picks = P.confirmation(scores, all_pools, joint)
            self.assertTrue(8 <= len(confirmed) <= 12)
            for a in confirmed:
                self.assertTrue(set(P.subset_closure(a, all_pools)) <= set(confirmed))
            self.assertIn(picks['best_single_pilot'], confirmed)
            self.assertIn(picks['best_three_pilot'], confirmed)

    def test_interaction_contrasts_are_zero_for_additive_effects(self):
        choices = {str(s): [f'{f}_s', f'{f}_si'] for s, f in enumerate(P.FAMILIES)}
        pools = P.joint_matrix(self.pools, choices)
        scores = {a: [metric(.9 + .001 * P.slot_count(p, pools['control']))] for a, p in pools.items()}
        for r in P.interaction_rows(scores, pools, [0]):
            self.assertAlmostEqual(r['Nonadditivity'], 0.)
            self.assertAlmostEqual(r['FactorialInteraction'], 0.)
        scores['j_s_s_s'] = [metric(.905)]
        row = next(r for r in P.interaction_rows(scores, pools, [0]) if r['Arm'] == 'j_s_s_s' and r['Metric'] == 'DSC')
        self.assertAlmostEqual(row['Nonadditivity'], .002)
        self.assertAlmostEqual(row['FactorialInteraction'], .002)

    def test_new_selection_chain_preserves_triples_even_when_all_lose(self):
        write(self.root / 'manifest.json', {'families': self.groups, 'pools': self.pools})
        S.make_plan(self.root, 'pilot', 'pilot', self.pools, [0])
        local = {a: [metric(.89 if a != 'control' else .9)] for a in self.pools}
        write(self.root / 'full/test-38/scores.json', {'poison': [metric(.999)]})
        with patch.object(S, 'checked_report', return_value=(local, {})):
            S.select(self.root, 'pilot'); S.select(self.root, 'pilot')
        joint_plan = S.get_plan(self.root, 'joint')
        self.assertEqual(joint_plan['phase'], 'pilot')
        joint = {a: local.get(a, [metric(.89)]) for a in joint_plan['pools']}
        with patch.object(S, 'checked_report', side_effect=lambda root, stage: (local if stage == 'pilot' else joint, {})):
            S.select(self.root, 'joint'); S.select(self.root, 'joint')
        full = S.get_plan(self.root, 'confirm')
        self.assertTrue(8 <= len(full['pools']) <= 12)
        scores = {a: [metric(.89 if a != 'control' else .9)] * 3 for a in full['pools']}
        with patch.object(S, 'checked_report', return_value=(scores, {})):
            S.select(self.root, 'confirm'); S.select(self.root, 'confirm')
        selected = P.read(self.root / 'selection.json')
        self.assertEqual(selected['selected_overall'], 'control')
        self.assertFalse(selected['qualified_multi_slot_combination'])
        self.assertEqual(set(selected['test_arms']), set(full['pools']))
        self.assertEqual(P.slot_count(full['pools'][selected['best_three_diagnostic']], full['pools']['control']), 3)
        S.get_plan(self.root, 'test', deep=True)
        selected['selected_overall'] = 'poison'
        write(self.root / 'selection.json', selected)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            S.get_plan(self.root, 'test')


    def test_final_combination_report_compares_constituents_and_marks_selection(self):
        import csv
        joint = P.joint_matrix(self.pools, {str(s): [f'{f}_e', f'{f}_si'] for s, f in enumerate(P.FAMILIES)})
        pools = {a: joint[a] for a in P.subset_closure('j_e_e_e', joint)}
        cfg = {'border_points': 1000}
        write(self.root / 'manifest.json', {'case_ids': {'test-38': ['A_002']}, 'student_config': cfg})
        write(self.root / 'reference-images.json', {'algorithm': 'sha256:' + 'a' * 64})
        write(self.root / 'selection.json', {'selected_overall': 'j_e_e_e', 'best_combination': 'j_e_e_e',
                                           'qualified_multi_slot_combination': True, 'best_three_diagnostic': 'j_e_e_e', 'best_single_diagnostic': 'v2_e'})
        (self.root / 'pretrained-frozen.pth').write_bytes(b'official-fixture')
        S.make_plan(self.root, 'test', 'full', pools, [0, 1, 2])
        entries = [(-1, 'pretrained')] + [(s, a) for s in range(3) for a in pools]
        for seed, arm in entries:
            if arm != 'pretrained':
                (self.root / f'full/train/seed_{seed}/{arm}/cotracker_three_final.pth').write_bytes(arm.encode())
            S.bind_evaluation(self.root, 'test', 'test-38', seed, arm)
            relative = 'pretrained' if arm == 'pretrained' else f'seed_{seed}/{arm}'
            folder = self.root / 'full/test-38' / relative
            values = dict(zip(S.L.FIELDS.values(), (.902 if arm == 'j_e_e_e' else .901, 4., 1.5, 1.7, .9)))
            write(folder / P.PROFILE / 'metrics.json', {'aggregates': {**values, 'total_time': 200.},
                   'results': [{'case_id': 'A_002', **values}]})
            write(folder / 'frozen-images.json', P.read(self.root / 'reference-images.json'))
            job = folder / P.PROFILE / 'checkpoint/jobs/A_002'
            output = job / 'output/images/mri-linac-series-targets/output.mha'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'output-fixture')
            (job / '.complete').write_text('done')
            write(job / 'diagnostics.json', {'profile': P.PROFILE, 'config': cfg,
                  'model': {'parameters': 25385700 if arm == 'pretrained' else 28763025,
                            'feature_sidecar': 'none' if arm == 'pretrained' else 'mamba',
                            'frozen_pretrained': True, 'sidecar_runtime': P.RUNTIME},
                  'prediction': {'output_file_sha256': P.sha(output)},
                  'mechanism': {'sidecar_calls': 2, 'sidecar_residual_rms_max': .01}})
        with patch.object(S, 'checked_training', return_value={'artifacts_sha256': {}}):
            S.report(self.root, 'test', 'test-38')
        target = self.root / 'reports/test/test-38'
        with (target / 'paired-deltas.csv').open(encoding='utf-8-sig') as f:
            paired = list(csv.DictReader(f))
        self.assertEqual({r['Reference'] for r in paired if r['Candidate'] == 'j_e_e_e'},
                         {'control', 'pretrained', 'v2_e', 'on3_e', 'offscaled_e', 'j_e_e_0', 'j_e_0_e', 'j_0_e_e'})
        with (target / 'summary.csv').open(encoding='utf-8-sig') as f:
            summary = list(csv.DictReader(f))
        chosen = next(r for r in summary if r['Arm'] == 'j_e_e_e')
        self.assertEqual(chosen['N'], '3')
        self.assertIn('validation_selected_combination', chosen['SelectionRole'])
        self.assertIn('validation_selected_overall', chosen['SelectionRole'])

        with (target / 'interactions.csv').open(encoding='utf-8-sig') as f:
            interactions = list(csv.DictReader(f))
        row = next(r for r in interactions if r['Arm'] == 'j_e_e_e' and r['Metric'] == 'DSC')
        self.assertAlmostEqual(float(row['FactorialInteraction']), .001)
        hashes = P.read(target / 'audit.json')['artifacts_sha256']
        self.assertIn('reports/test/test-38/interactions.csv', hashes)


if __name__ == '__main__':
    unittest.main()
