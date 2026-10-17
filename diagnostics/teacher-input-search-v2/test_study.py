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

    def test_matrix_single_factors_and_budget(self):
        self.assertEqual(len(self.pools), 18)
        self.assertEqual(len({P.digest(p) for p in self.pools.values()}), 18)
        for family, names in self.groups.items():
            parent = self.pools[names[0]]
            self.assertEqual(len(names), 5)
            for name in names[1:]:
                changes = [(slot, k) for slot in range(3) for k in parent[slot] if parent[slot][k] != self.pools[name][slot][k]]
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0][0], P.FAMILY_SLOT[family])
        confirm = {p: self.pools[p] for p in ('control', 'oldpool', 'v2_flip_x', 'on3_flip_y', 'offold_percentile', 'offscaled_gamma')}
        combos = P.combinations(confirm, {0: 'v2_flip_x', 1: 'on3_flip_y', 2: 'offscaled_gamma'})
        self.assertEqual(len(combos), 8)
        self.assertEqual(len(set(confirm) | set(combos)), 10)  # at most 30 full runs, including reusable controls

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

    def test_combinations_do_not_replace_the_same_offline_slot_twice(self):
        winners = {0: 'v2_flip_x', 1: 'control', 2: 'offold_gamma'}
        pools = P.combinations(self.pools, winners)
        self.assertEqual(len(pools), 4)
        self.assertEqual(pools['mix02'][2]['checkpoint'], 'baseline_offline.pth')
        self.assertEqual(pools['mix02'][1], self.pools['control'][1])
        self.assertEqual(P.combinations(self.pools, {0: 'control', 1: 'control', 2: 'control'}), {'control': self.pools['control']})

    def test_all_transforms_preserve_time_rng_dtype_and_student_input(self):
        video = torch.arange(1 * 60 * 3 * 32 * 48, dtype=torch.float32).reshape(1, 60, 3, 32, 48).remainder(251)
        queries = torch.tensor([[[0., 0., 0.], [29., 22., 23.], [59., 47., 31.]]])
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
                expected_count = (25, 0, 100)[slot]
                self.assertEqual(model.seen[0].shape[1], 3 + expected_count)
                self.assertEqual(audit['teacher_calls'], 1)
                self.assertEqual(audit['transform']['mode'], self.pools[name][slot]['transform'])
                self.assertEqual(audit['pool_sha256'], P.digest(self.pools[name]))
                self.assertEqual(p.shape, (1, 60, 3, 2))
                offset = .4 + (3 + expected_count) / 1000
                expected = queries[..., 1:].unsqueeze(1).repeat(1, 60, 1, 1) + offset
                if audit['transform']['mode'] == 'flip_x':
                    expected[..., 0] -= 2 * offset
                if audit['transform']['mode'] == 'flip_y':
                    expected[..., 1] -= 2 * offset
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

    def test_intensity_uses_one_clip_range_and_flat_fallback(self):
        video = torch.linspace(10, 100, 48).reshape(1, 2, 1, 4, 6)
        q = torch.tensor([[[1., 3., 2.]]])
        norm, qq, _, meta = P.transform_inputs(video, q, q[:, :0], 'percentile')
        expected = ((video - meta['low']) / (meta['high'] - meta['low'])).clamp(0, 1) * 255
        self.assertTrue(torch.equal(norm, expected))
        self.assertEqual(norm.dtype, torch.float32)
        flat = torch.ones_like(video) * 42
        result = P.transform_inputs(flat, q, q[:, :0], 'percentile')
        self.assertTrue(result[3]['fallback'] and torch.equal(result[0], flat))
        gamma = P.transform_inputs(video, q, q[:, :0], 'gamma')[0]
        self.assertTrue(torch.equal(gamma, (video / 255).pow(.8) * 255))
        for bad in (video.double(), video - 500, video * float('nan')):
            with self.assertRaises(ValueError):
                P.transform_inputs(bad, q, q[:, :0], 'gamma')
        with self.assertRaises(ValueError):
            P.transform_inputs(video, q, q[:, :0], 'reverse')

    def test_baseline_equals_previous_selected_pool(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('prior_search_protocol', REPO / 'diagnostics/teacher-pool-search-v1/protocol.py')
        prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
        expected = prior.matrix()[0]['on3_s0']
        actual = [{k: v for k, v in s.items() if k != 'transform'} for s in P.baseline()]
        self.assertEqual(actual, expected)
        self.assertEqual(self.pools['oldpool'][1]['support'], 0)

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
        for arm in ('v2_flip_x', 'on3_flip_y', 'offold_percentile', 'offscaled_gamma'):
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
                    ('confirm', 'full', {'control': self.pools['control'], 'oldpool': self.pools['oldpool']}, [0, 1, 2])):
            S.make_plan(self.root, stage, phase, pools, seeds)
            S.get_plan(self.root, stage)
        pilot = P.read(self.root / 'pilot/train/seed_0/control/run-config.json')
        full = P.read(self.root / 'full/train/seed_0/control/run-config.json')
        self.assertEqual((pilot['steps'], full['steps']), (250, 1000))
        self.assertEqual(pilot['scheduler_num_steps'], full['scheduler_num_steps'])
        before = P.sha(self.root / 'full/train/seed_0/control/run-config.json')
        S.make_plan(self.root, 'combine', 'full', {'control': self.pools['control']}, [0, 1, 2])
        self.assertEqual(before, P.sha(self.root / 'full/train/seed_0/control/run-config.json'))
        changed = copy.deepcopy(self.pools['control']); changed[2]['iterations'] = 8
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            S.make_plan(self.root, 'combine', 'full', {'control': changed}, [0, 1, 2])

    def test_selection_chain_never_reads_test_results(self):
        manifest = {'families': self.groups, 'pools': self.pools}
        write(self.root / 'manifest.json', manifest)
        pilot = {a: [metric(.9)] for a in self.pools}
        for a in ('v2_flip_x', 'on3_flip_y', 'offold_percentile', 'offscaled_gamma'):
            pilot[a] = [metric(.902)]
        # Deliberately put absurdly high values in a test file: it must not be used.
        write(self.root / 'full/test-38/scores.json', {'control': [metric(.999)]})
        with patch.object(S, 'checked_report', return_value=(pilot, {})):
            S.select(self.root, 'pilot')
            S.select(self.root, 'pilot')
        confirm_plan = S.get_plan(self.root, 'confirm')
        scores = {a: [metric(.9)] * 3 for a in confirm_plan['pools']}
        for a in ('v2_flip_x', 'on3_flip_y', 'offscaled_gamma'):
            scores[a] = [metric(.902)] * 3
        with patch.object(S, 'checked_report', return_value=(scores, {})):
            S.select(self.root, 'confirm')
            S.select(self.root, 'confirm')
        combo = S.get_plan(self.root, 'combine')
        scores = {a: [metric(.901)] * 3 for a in combo['pools']}
        scores['control'] = [metric(.9)] * 3
        scores['mix012'] = [metric(.904)] * 3
        with patch.object(S, 'checked_report', return_value=(scores, {})):
            S.select(self.root, 'combine')
            S.select(self.root, 'combine')
        selection = P.read(self.root / 'selection.json')
        self.assertEqual(selection['selected_overall'], 'mix012')
        self.assertTrue(selection['qualified_multi_slot_combination'])
        self.assertIn('oldpool', selection['test_arms'])
        self.assertEqual(S.get_plan(self.root, 'test')['phase'], 'full')
        selection['selected_overall'] = 'control'
        write(self.root / 'selection.json', selection)
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            S.get_plan(self.root, 'test')

    def test_no_gain_combination_falls_back_without_forced_replacement(self):
        write(self.root / 'manifest.json', {})
        S.make_plan(self.root, 'combine', 'full', {a: self.pools[a] for a in ('control', 'oldpool')}, [0, 1, 2])
        with patch.object(S, 'checked_report', return_value=({'control': [metric()] * 3, 'oldpool': [metric(.89)] * 3}, {})):
            S.select(self.root, 'combine')
        selection = P.read(self.root / 'selection.json')
        self.assertEqual(selection['selected_overall'], 'control')
        self.assertFalse(selection['qualified_multi_slot_combination'])

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
            self.assertEqual(len(list(out.glob('*/train/seed_*/*/run-config.json'))), 23)
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
        plan = S.make_plan(self.root, 'pilot', 'pilot', {'control': self.pools['control'], 'v2_flip_x': self.pools['v2_flip_x']}, [0])
        for arm, kind, score, seed in (('pretrained', 'none', .9, -1), ('control', 'mamba', .901, 0), ('v2_flip_x', 'mamba', .902, 0)):
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
            self.assertEqual(float(next(r for r in rows if r['Arm'] == 'v2_flip_x')['DSC_Mean']), .902)
            S.checked_report(self.root, 'pilot')
            with self.assertRaisesRegex(ValueError, 'Only frozen'):
                S.report(self.root, 'pilot', 'test-38')
            checkpoint = self.root / 'pilot/train/seed_0/v2_flip_x/cotracker_three_final.pth'
            checkpoint.write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError, 'Frozen'):
                S.bind_evaluation(self.root, 'pilot', 'validation-10', 0, 'v2_flip_x')
            with self.assertRaisesRegex(ValueError, 'checkpoint binding'):
                S.report(self.root, 'pilot', 'validation-10')

    def test_final_combination_report_compares_constituents_and_marks_selection(self):
        import csv
        pools = P.combinations(self.pools, {0: 'v2_flip_x', 1: 'on3_flip_y', 2: 'control'})
        pools['oldpool'] = self.pools['oldpool']
        cfg = {'border_points': 1000}
        write(self.root / 'manifest.json', {'case_ids': {'test-38': ['A_002']}, 'student_config': cfg})
        write(self.root / 'reference-images.json', {'algorithm': 'sha256:' + 'a' * 64})
        write(self.root / 'selection.json', {'selected_overall': 'mix01', 'best_combination': 'mix01',
                                           'qualified_multi_slot_combination': True})
        (self.root / 'pretrained-frozen.pth').write_bytes(b'official-fixture')
        S.make_plan(self.root, 'test', 'full', pools, [0, 1, 2])
        entries = [(-1, 'pretrained')] + [(s, a) for s in range(3) for a in pools]
        for seed, arm in entries:
            if arm != 'pretrained':
                (self.root / f'full/train/seed_{seed}/{arm}/cotracker_three_final.pth').write_bytes(arm.encode())
            S.bind_evaluation(self.root, 'test', 'test-38', seed, arm)
            relative = 'pretrained' if arm == 'pretrained' else f'seed_{seed}/{arm}'
            folder = self.root / 'full/test-38' / relative
            values = dict(zip(S.L.FIELDS.values(), (.902 if arm == 'mix01' else .901, 4., 1.5, 1.7, .9)))
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
        self.assertEqual({r['Reference'] for r in paired if r['Candidate'] == 'mix01'},
                         {'control', 'oldpool', 'pretrained', 'v2_flip_x', 'on3_flip_y'})
        with (target / 'summary.csv').open(encoding='utf-8-sig') as f:
            summary = list(csv.DictReader(f))
        chosen = next(r for r in summary if r['Arm'] == 'mix01')
        self.assertEqual(chosen['N'], '3')
        self.assertIn('validation_selected_combination', chosen['SelectionRole'])
        self.assertIn('validation_selected_overall', chosen['SelectionRole'])

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


if __name__ == '__main__':
    unittest.main()
