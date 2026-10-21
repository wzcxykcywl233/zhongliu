"""CPU mechanism, provenance and adapter tests; not a substitute for GPU smoke."""
import ast
import copy
import json
import os
from pathlib import Path
import random
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
import numpy as np
import torch
from cotracker.models.core.cotracker.losses import sequence_loss, huber_loss
from cotracker.models.core.model_utils import reduce_masked_mean
from cotracker.utils.offline_teacher_policy import tensor_sha


class Teacher(torch.nn.Module):
    def __init__(self, offset=0., consume_rng=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        self.consume_rng = consume_rng
        self.eval()

    def forward(self, video, queries, iters=4):
        if self.consume_rng:
            random.random(); np.random.random(); torch.rand(3)
        shape = (1, video.shape[1], queries.shape[1])
        p = queries[..., 1:].unsqueeze(1).repeat(1, shape[1], 1, 1) + self.weight
        return p, torch.full(shape, .95), torch.full(shape, .8), None


def metric(dsc=.9, d98=.9, hd95=4.):
    return dict(DSC=dsc, D98=d98, HD95=hd95, MASD=1.5, CD=1.7, TimeSec=200.)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def context(self, arm, offsets=(3., 3.5), frames=4, points=8, time=1):
        q = torch.tensor([[[float(time), 10., 10.]]]).repeat(1, points, 1)
        p = torch.full((1, frames, points, 2), 10.)
        v = torch.full((1, frames, points), .95)
        c = torch.full_like(v, .8)
        reviewers = [(p + torch.tensor([d, 0.]), v.clone(), c.clone()) for d in offsets]
        co = P.Coordinator(arm, P.matrix()[arm])
        co.context = dict(primary=(p, v, c), reviewers=[] if arm == 'control' else reviewers,
                          queries=q, height=64, width=64, slot=0, audit={})
        return co

    def loss(self, co, validity=None):
        p, v, _ = co.context['primary']
        pred = (p + torch.linspace(.1, 2., p.numel()).reshape_as(p)).requires_grad_()
        value = co.loss(sequence_loss, [[pred, pred + .1]], [p],
                        [torch.ones_like(v) if validity is None else validity], vis=[(v > .9).float()],
                        add_huber_loss=True, loss_only_for_visible=True)
        value.backward()
        return value.detach(), pred.grad, co.context['audit']

    def test_matrix_keeps_teacher_and_student_configs_identical(self):
        self.assertEqual(len(P.matrix()), 8)
        for arm, pool in P.matrix().items():
            cfg = P.config(0, arm, 'smoke', pool, 'a' * 64)
            self.assertEqual(pool['teachers'], P.B.baseline())
            self.assertEqual(cfg['sequence_len'], 60)
            self.assertEqual(cfg['traj_per_sample'], 128)
            self.assertEqual(cfg['auxiliary_teacher_weight'], 0)
        with self.assertRaises(ValueError):
            P.config(0, 'shadow', 'full', P.matrix()['shadow'], 'a' * 64)

    def test_dispatch_preserves_primary_labels_input_and_rng_all_slots(self):
        video = torch.zeros(1, 60, 3, 64, 64)
        q = torch.tensor([[[0., 10., 10.], [29., 20., 20.], [59., 30., 30.]]])
        for slot in range(3):
            models = [(typ, Teacher(float(i), True)) for i, typ in enumerate(P.TEACHER_TYPES)]
            draw = P.support_for(slot, video, q)
            results = []
            for arm in P.matrix():
                random.seed(99); np.random.seed(99); torch.manual_seed(99)
                co = P.Coordinator(arm, P.matrix()[arm])
                out = co.dispatch(slot, *models[slot], video, q, draw, arm, models)
                state = (random.random(), float(np.random.random()), torch.rand(3))
                results.append((out, state))
                self.assertTrue(torch.equal(video, torch.zeros_like(video)))
                self.assertEqual(out[3]['teacher_calls'], 1 if arm in ('control', 'same_teacher') else 3)
            for out, state in results[1:]:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(out[:3], results[0][0][:3])))
                self.assertEqual(state[:2], results[0][1][:2])
                self.assertTrue(torch.equal(state[2], results[0][1][2]))
            for typ, m in models:
                self.assertFalse(m.training)
                self.assertIsNone(m.weight.grad)

    def test_wrong_teacher_mode_or_identity_fails(self):
        video = torch.zeros(1, 60, 3, 64, 64)
        q = torch.tensor([[[0., 10., 10.]]])
        models = [(typ, Teacher()) for typ in P.TEACHER_TYPES]
        models[0][1].train()
        co = P.Coordinator('gate2', P.matrix()['gate2'])
        with self.assertRaisesRegex(ValueError, 'unfrozen'):
            co.dispatch(0, *models[0], video, q, P.support_for(0, video, q), 'gate2', models)

    def test_exact_noop_loss_and_gradients(self):
        reference = self.loss(self.context('control'))
        for arm in ('shadow', 'same_teacher'):
            out = self.loss(self.context(arm, offsets=(0., 0.) if arm == 'same_teacher' else (3., 3.5)))
            self.assertTrue(torch.equal(out[0], reference[0]))
            self.assertTrue(torch.equal(out[1], reference[1]))

    def test_gate_requires_two_reviewers_not_just_one(self):
        _, _, q = self.loss(self.context('gate2'))
        self.assertEqual(q['conflict'], 16)
        self.assertEqual(q['weight_mass'], 20.)
        for offsets in ((3., -3.), (3., 0.), (2., 2.), (3., 6.)):
            _, _, q = self.loss(self.context('gate2', offsets=offsets))
            self.assertEqual(q['downweighted'], 0)
        _, _, q4 = self.loss(self.context('gate4'))
        self.assertEqual(q4['downweighted'], 0)

    def test_gate_excludes_query_and_before_query(self):
        co = self.context('gate2')
        _, grad, q = self.loss(co)
        self.assertEqual(q['eligible'], 16)
        # No trajectory target is changed even on the downweighted frames.
        self.assertEqual(q['target_sha256'], tensor_sha(co.context['primary'][0]))
        self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_bad_reviewers_abstain_without_nan(self):
        for failure in ('nan', 'oob', 'invisible', 'vnan'):
            co = self.context('gate2')
            p, v, c = co.context['reviewers'][0]
            if failure == 'nan': p.fill_(float('nan'))
            elif failure == 'oob': p.fill_(100.)
            elif failure == 'invisible': v.fill_(.9)
            else: v.fill_(float('nan'))
            val, grad, q = self.loss(co)
            self.assertEqual(q['eligible'], 0)
            self.assertEqual(q['downweighted'], 0)
            self.assertTrue(bool(torch.isfinite(val)) and bool(torch.isfinite(grad).all()))

    def test_fusion_small_and_only_three_way_agreement(self):
        co = self.context('fuse2', offsets=(2., 2.))
        _, _, q = self.loss(co)
        self.assertEqual(q['fused'], 16)
        self.assertEqual(q['max_correction'], .5)
        expected = co.context['primary'][0].clone()
        expected[:, 2:, :, 0] += .5
        self.assertEqual(q['target_sha256'], tensor_sha(expected))
        _, _, q = self.loss(self.context('fuse2'))
        self.assertEqual(q['fused'], 0)
        self.assertEqual(q['downweighted'], 16)

    def test_shuffle_is_count_and_mass_matched_not_global_rng(self):
        outputs = []
        for arm in ('gate2', 'shuffle2', 'shuffle2'):
            co = self.context(arm, points=32)
            # Only half the points conflict; the other half agree with primary.
            for p, _, _ in co.context['reviewers']:
                p[:, :, 16:] = co.context['primary'][0][:, :, 16:]
            rng = torch.get_rng_state().clone()
            out = self.loss(co)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            outputs.append(out[2])
        for q in outputs[1:]:
            for k in ('eligible', 'conflict', 'downweighted', 'base_weight_mass', 'weight_mass'):
                self.assertEqual(q[k], outputs[0][k])
        self.assertNotEqual(outputs[0]['weights_sha256'], outputs[1]['weights_sha256'])
        self.assertEqual(outputs[1]['weights_sha256'], outputs[2]['weights_sha256'])

    def test_tutor_matches_per_point_convex_mixture(self):
        co = self.context('tutor010', offsets=(1., 2.), time=0)
        p, v, _ = co.context['primary']
        pred = (p + .5).requires_grad_()
        got = co.loss(sequence_loss, [[pred]], [p], [torch.ones_like(v)], vis=[(v > .9).float()],
                      add_huber_loss=True, loss_only_for_visible=True)
        main = huber_loss(pred, p, 6.).mean(-1)
        expected = main.clone()
        expected[:, 1:] = .9 * main[:, 1:]
        for r in co.context['reviewers']:
            expected[:, 1:] += .05 * huber_loss(pred, r[0], 6.).mean(-1)[:, 1:]
        self.assertTrue(torch.allclose(got, reduce_masked_mean(expected, torch.ones_like(v))))
        co.context['reviewers'][0][1].fill_(0.)
        val, grad, _ = self.loss(co)
        self.assertTrue(bool(torch.isfinite(val)) and bool(torch.isfinite(grad).all()))

    def test_empty_validity_and_no_agreement_match_baseline(self):
        reference = self.loss(self.context('control'))
        for arm in P.ARMS:
            co = self.context(arm)
            for _, v, _ in co.context['reviewers']: v.zero_()
            val, grad, _ = self.loss(co)
            self.assertTrue(torch.equal(val, reference[0]))
            self.assertTrue(torch.equal(grad, reference[1]))
            val, grad, _ = self.loss(co, torch.zeros_like(co.context['primary'][1]))
            self.assertEqual(float(val), 0.)
            self.assertEqual(int(grad.count_nonzero()), 0)

    def test_nonbinary_validity_is_rejected(self):
        co = self.context('gate2')
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.loss(co, torch.full_like(co.context['primary'][1], .5))

    def fake_rows(self, arm='gate2'):
        cfg = P.config(0, arm, 'smoke', P.matrix()[arm], 'a' * 64)
        video = torch.zeros(1, 60, 3, 64, 64)
        q = torch.tensor([[[0., 10., 10.]]]).repeat(1, 128, 1)
        models = [(typ, Teacher(float(slot))) for slot, typ in enumerate(P.TEACHER_TYPES)]
        sampler = S.L.TeacherPairSampler(3, seed=P.BASE_SEED)
        rows = []
        for i in range(10):
            slot = sampler.sample().primary
            co = P.Coordinator(arm, P.matrix()[arm])
            p, v, c, audit = co.dispatch(slot, *models[slot], video, q, P.support_for(slot, video, q), arm, models)
            val, _, _ = self.loss(co)
            rows.append(dict(step=i, architecture='base', teacher_policy=arm, label_audit=audit,
                case_names=['A_001'], queries_sha256=tensor_sha(q), video_sha256=tensor_sha(video),
                primary_index=slot, auxiliary_index=None, loss=float(val), step_seconds=2.,
                health=dict(peak_allocated_mib=12000., peak_reserved_mib=13000., driver_free_mib=10000.)))
        return cfg, rows

    def parse_rows(self, cfg, rows):
        path = self.root / 'paired.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return P.records(path, cfg)

    def test_record_replay_valid_and_reviewer_change_rejected(self):
        cfg, rows = self.fake_rows()
        self.parse_rows(cfg, rows + [copy.deepcopy(rows[0])])
        bad = copy.deepcopy(rows[0])
        bad['label_audit']['collaboration']['reviewers'][0]['tracks_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'Resumed'):
            self.parse_rows(cfg, rows + [bad])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.parse_rows(cfg, rows[:-1])

    def test_pair_audit_primary_labels_and_queries_are_immutable(self):
        a, ar = self.fake_rows('control'); b, br = self.fake_rows('gate2')
        aa, bb = self.parse_rows(a, ar), self.parse_rows(b, br)
        P.paired(aa, bb, a['pool'], b['pool'])
        bb[0]['label_audit']['tracks_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'Primary'):
            P.paired(aa, bb, a['pool'], b['pool'])

    def test_audit_rejects_wrong_rule_memory_or_primary_labels(self):
        cfg, rows = self.fake_rows()
        for target, key, value in [('q', 'rule_sha256', 'f' * 64), ('q', 'student_forward_passes', 2),
                                    ('q', 'max_correction', 1.), ('a', 'base_visibility_sha256', 'f' * 64),
                                    ('h', 'driver_free_mib', 1000.)]:
            bad = copy.deepcopy(rows)
            d = dict(q=bad[0]['label_audit']['collaboration'], a=bad[0]['label_audit'], h=bad[0]['health'])[target]
            d[key] = value
            with self.assertRaises(ValueError): self.parse_rows(cfg, bad)

    def test_instrumentation_preserves_other_losses_and_single_student_forward(self):
        raw = (S.APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8')
        adapted = P.instrument(raw, S.legacy_protocol.instrument)
        tree = ast.parse(adapted)
        compile(tree, 'adapted', 'exec')
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_forward_batch_single_teacher')
        branch = next(x for x in fn.body if isinstance(x, ast.If) and ast.unparse(x.test) == "args.model_name == 'cotracker_three'")
        calls = [x for statement in branch.body for x in ast.walk(statement) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)]
        self.assertEqual(sum(x.func.id == 'model' for x in calls), 1)
        self.assertEqual(sum(x.func.id == '__coordinate_loss__' for x in calls), 1)
        self.assertEqual(sum(x.func.id == 'sequence_loss' for x in calls), 1)  # invisible loss
        self.assertEqual(sum(x.func.id == 'sequence_prob_loss' for x in calls), 1)
        self.assertIn('teacher_models)', adapted)
        with self.assertRaises(ValueError): P.instrument(raw.replace('    vis_g = vis_g > 0.9', '    vis_g = vis_g > 0.8'), S.legacy_protocol.instrument)

    def test_selection_fallback_and_three_seed_requirement(self):
        scores = {'control': [metric()] * 3, 'bad': [metric(.899)] * 3,
                  'd98_bad': [metric(.902, .89)] * 3, 'one_seed': [metric(.904), metric(.8999), metric(.8999)]}
        self.assertEqual(P.best(scores, ['bad', 'd98_bad', 'one_seed']), 'control')
        scores['good'] = [metric(.901), metric(.902), metric(.8999)]
        self.assertEqual(P.best(scores, ['good']), 'good')

    def test_fixed_full_matrix_and_validation_only_selection(self):
        plans = {'pools': {a: P.matrix()[a] for a in P.ARMS}, 'seeds': [0], 'phase': 'pilot'}
        scores = {a: [metric()] for a in P.ARMS}
        with patch.object(S, 'checked_report', return_value=(scores, {})), patch.object(S, 'get_plan', return_value=plans), patch.object(S, 'make_plan') as make:
            S.select(self.root, 'pilot')
            self.assertEqual(set(make.call_args.args[3]), set(P.ARMS))
            self.assertEqual(make.call_args.args[4], [0, 1, 2])
            scores = {a: [metric()] * 3 for a in P.ARMS}
            scores['shuffle2'] = [metric(.91)] * 3  # diagnostic not adoptable
        with patch.object(S, 'checked_report', return_value=(scores, {})), patch.object(S, 'get_plan', return_value=plans), patch.object(S, 'make_plan'):
            S.select(self.root, 'confirm')
            self.assertEqual(P.read(self.root / 'selection.json')['selected_overall'], 'control')
        self.assertIs(S.B.P, P)

    def test_real_model_audit_rejects_backbone_or_teacher_pool_changes(self):
        model = S.L.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        info = S.L.attach_sidecar(model, 'mamba')
        info.update(execution_runtime=P.RUNTIME, initialization_outputs_exact=True,
                    initialization_reference='frozen-pretrained', initialization_max_abs_difference=[0., 0., 0.])
        with torch.no_grad(): model.updateformer.feature_sidecar.output.weight.fill_(.001)
        write(self.root / 'manifest.json', {'trainer_sha256': 'a' * 64,
              'package_sha256': {'train.py': 'b' * 64, 'protocol.py': 'c' * 64}})
        write(self.root / 'reference-proof.json', {'base_tensor_sha256': info['base_tensor_sha256']})
        plan = S.make_plan(self.root, 'smoke', 'smoke', {'control': P.matrix()['control']}, [0])
        folder = self.root / 'smoke/train/seed_0/control'
        cfg, rows = self.fake_rows('control')
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
            with self.assertRaisesRegex(ValueError, 'pool'): S.verify_arm(self.root, plan, 0, 'control')
            final['teacher_pool_sha256'] = cfg['pool_sha256']
            key = next(k for k in final['model'] if '.feature_sidecar.' not in k)
            final['model'][key] = final['model'][key] + .01
            with self.assertRaisesRegex(ValueError, 'Frozen base'): S.verify_arm(self.root, plan, 0, 'control')

    def test_official_report_mechanism_deltas_and_checkpoint_binding(self):
        import csv
        cases = ['A_010', 'A_012']
        cfg = {'border_points': 1000}
        write(self.root / 'manifest.json', {'case_ids': {'validation-10': cases}, 'student_config': cfg})
        write(self.root / 'reference-images.json', {'algorithm': 'sha256:' + 'a' * 64})
        (self.root / 'pretrained-frozen.pth').write_bytes(b'official-fixture')
        S.make_plan(self.root, 'pilot', 'pilot', {a: P.matrix()[a] for a in P.ARMS}, [0])
        for i, arm in enumerate(('pretrained', *P.ARMS)):
            kind, seed = ('none', -1) if arm == 'pretrained' else ('mamba', 0)
            if kind != 'none':
                (self.root / f'pilot/train/seed_0/{arm}/cotracker_three_final.pth').write_bytes(arm.encode())
            S.B.bind_evaluation(self.root, 'pilot', 'validation-10', seed, arm)
            relative = 'pretrained' if arm == 'pretrained' else 'seed_0/' + arm
            folder = self.root / 'pilot/validation-10' / relative
            values = dict(zip(S.L.FIELDS.values(), (.9 + i * .001, 4., 1.5, 1.7, .9)))
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
        with patch.object(S, 'checked_training', return_value={'artifacts_sha256': {}}), patch.object(S.B, 'checked_training', return_value={'artifacts_sha256': {}}):
            S.report(self.root, 'pilot', 'validation-10')
            S.report(self.root, 'pilot', 'validation-10')
            target = self.root / 'reports/pilot/validation-10'
            with (target / 'summary.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 7)
            with (target / 'mechanism-deltas.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[0]['Delta_DSC']), -.002)
            S.checked_report(self.root, 'pilot')
            with self.assertRaisesRegex(ValueError, 'Only frozen'): S.report(self.root, 'pilot', 'test-38')
            checkpoint = self.root / 'pilot/train/seed_0/gate2/cotracker_three_final.pth'
            checkpoint.write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError, 'checkpoint binding'): S.report(self.root, 'pilot', 'validation-10')

    def test_modified_parent_results_block_downstream_plan(self):
        write(self.root / 'manifest.json', {})
        artifact = self.root / 'validation-scores.json'
        write(artifact, {'DSC': .9})
        proof = self.root / 'reports/pilot/validation-10/audit.json'
        write(proof, {'artifacts_sha256': {'validation-scores.json': P.sha(artifact)}})
        parent = {proof.relative_to(self.root).as_posix(): P.sha(proof)}
        S.make_plan(self.root, 'confirm', 'full', {'control': P.matrix()['control']}, [0, 1, 2], parent)
        S.get_plan(self.root, 'confirm', deep=True)
        write(artifact, {'DSC': .99})
        with self.assertRaisesRegex(ValueError, 'artifact changed'): S.get_plan(self.root, 'confirm', deep=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
