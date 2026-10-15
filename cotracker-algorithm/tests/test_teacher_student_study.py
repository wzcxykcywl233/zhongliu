"""CPU policy, RNG, replay and paired-audit regressions. GPU training is remote."""
import copy
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ext/co-tracker'), str(ROOT / 'experiments')]
from cotracker.utils.offline_teacher_policy import (
    POLICIES, SETTINGS, offline_teacher_forward, label_audit, tensor_sha,
)
import teacher_student_study as study


class DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.eval()
        self.calls = []

    def forward(self, video, queries, iters=4):
        self.calls.append((queries.clone(), iters, torch.is_grad_enabled()))
        b, t = video.shape[:2]
        p = queries[:, None, :, 1:].expand(b, t, -1, -1) + self.offset
        v = torch.full(p.shape[:-1], .95)
        return p, v, v * .9, None


class TeacherPolicyTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.zeros(1, 10, 3, 16, 16)
        self.queries = torch.tensor([[[0., 3., 4.], [2., 5., 6.], [9., 7., 8.]]])
        self.support = torch.zeros(1, 100, 3)

    def test_all_policies_preserve_arbitrary_time_xy_order(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                teacher = DummyTeacher()
                before = self.queries.clone()
                result = offline_teacher_forward(teacher, self.video, self.queries, self.support, policy)
                received, iterations, grad = teacher.calls[0]
                self.assertTrue(torch.equal(received[:, :3], before))
                self.assertEqual(received.shape[1], 3 + SETTINGS[policy][1])
                self.assertEqual(iterations, SETTINGS[policy][2])
                self.assertFalse(grad)
                self.assertTrue(torch.equal(self.queries, before))
                self.assertEqual(result[0].shape, (1, 10, 3, 2))

    def test_no_random_draws_in_adapter(self):
        for policy in POLICIES:
            torch.manual_seed(43)
            before = torch.get_rng_state()
            offline_teacher_forward(DummyTeacher(), self.video, self.queries, self.support, policy)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_rejects_unfrozen_or_training_teacher(self):
        for teacher in (DummyTeacher().train(), DummyTeacher().requires_grad_(True)):
            with self.assertRaisesRegex(ValueError, 'frozen/eval'):
                offline_teacher_forward(teacher, self.video, self.queries, self.support, 'old')

    def test_rejects_invalid_times_instead_of_clamping(self):
        for bad in (-1., 10., 1.5, float('nan')):
            query = self.queries.clone()
            query[0, 1, 0] = bad
            with self.assertRaises(ValueError):
                offline_teacher_forward(DummyTeacher(), self.video, query, self.support, 'old')

    def test_labels_checked_before_visibility_binarization(self):
        result = offline_teacher_forward(DummyTeacher(), self.video, self.queries, self.support, 'grid0')
        audit = label_audit(self.video, self.queries, *result, policy='grid0',
                            teacher_type='offline_cotracker_three', support_draw_sha256='a'*64, seconds=.2)
        self.assertEqual(sum(audit['query_time_counts']), 3)
        self.assertEqual(audit['query_time_counts'][9], 1)
        self.assertEqual(audit['support_used'], 0)
        self.assertEqual(audit['visibility_sha256'], tensor_sha(result[1]))
        for index, bad in ((0, float('nan')), (1, 1.01), (2, -.1)):
            values = [x.clone() for x in result]
            values[index].flatten()[0] = bad
            with self.assertRaises(ValueError):
                label_audit(self.video, self.queries, *values, policy='old',
                            teacher_type='offline_cotracker_three', support_draw_sha256='a'*64, seconds=.2)

    def test_real_cpu_offline_model_adapter_matches_direct(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60).eval().requires_grad_(False)
            video = torch.rand(1, 3, 3, 64, 64) * 255
            queries = torch.tensor([[[0., 16., 16.], [1., 32., 32.], [2., 48., 48.]]])
            with torch.no_grad():
                actual = offline_teacher_forward(model, video, queries, self.support, 'grid0')
                direct = model(video, queries, iters=4)[:3]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, direct)))
        finally:
            torch.set_num_threads(previous)

    def test_real_trainer_teacher_prefix_legacy_parity_and_audit(self):
        # Execute the ACTUAL trainer teacher-label block, stopping before the
        # student loss. This does not import Lightning/CUDA on the CPU host.
        source = ROOT / 'ext/co-tracker/train_on_real_data.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_batch_single_teacher')
        cut = next(i for i, n in enumerate(function.body) if isinstance(n, ast.If) and
                   ast.unparse(n.test) == "args.model_name == 'cotracker_three'")
        function.body = function.body[:cut] + ast.parse('return batch').body
        from cotracker.models.core.model_utils import get_uniformly_sampled_pts, get_points_on_a_grid
        namespace = dict(torch=torch, time=time, offline_teacher_forward=offline_teacher_forward,
                         label_audit=label_audit, tensor_sha=tensor_sha,
                         get_uniformly_sampled_pts=get_uniformly_sampled_pts, get_points_on_a_grid=get_points_on_a_grid)
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, str(source), 'exec'), namespace)
        forward = namespace['_forward_batch_single_teacher']
        def run(study_enabled, teacher_type, policy):
            batch = SimpleNamespace(video=self.video, trajectory=torch.ones(1,10,3,2),
                                    visibility=torch.ones(1,10,3), valid=torch.ones(1,10,3))
            args = SimpleNamespace(teacher_student_study=study_enabled, offline_teacher_policy=policy,
                                   train_grid_size=2, train_sift_size=0)
            torch.manual_seed(79)
            with patch.object(torch.cuda, 'synchronize'):
                result = forward(batch, None, args, [(teacher_type, DummyTeacher())], 0, queries=self.queries)
            return result, torch.get_rng_state()
        for teacher_type in ('online', 'online_cotracker_three', 'offline_cotracker_three'):
            legacy, rng_legacy = run(False, teacher_type, 'old')
            current, rng_current = run(True, teacher_type, 'old')
            self.assertTrue(torch.equal(legacy.trajectory, current.trajectory))
            self.assertTrue(torch.equal(legacy.visibility, current.visibility))
            self.assertTrue(torch.equal(rng_legacy, rng_current))
            self.assertEqual(current.teacher_label_audit['query_time_counts'][9], 1)
        for policy in ('scaled','grid0','iter6'):
            _, rng = run(True, 'offline_cotracker_three', policy)
            _, old_rng = run(True, 'offline_cotracker_three', 'old')
            self.assertTrue(torch.equal(rng, old_rng))


def records(policy):
    rows = []
    for i, slot in enumerate((1, 0, 1, 0, 2)):
        changed = slot == 2 and policy != 'old'
        rows.append({'step': i, 'teacher_policy': policy, 'architecture': 'base',
                     'case_names': ['B_012'], 'queries_sha256': 'q'*64, 'video_sha256': 'v'*64,
                     'primary_index': slot, 'auxiliary_index': None, 'loss': .01, 'step_seconds': .6,
                     'label_audit': {'policy': policy, 'teacher_type': ('online','online_cotracker_three','offline_cotracker_three')[slot],
                                     'query_time_counts': [100,100,184,0,0,0,0,0,0,0],
                                     'support_draw_sha256': 'a'*64, 'tracks_sha256': ('c' if changed else 'b')*64,
                                     'visibility_sha256': 'b'*64, 'confidence_sha256': 'b'*64,
                                     'teacher_seconds': .1, 'visible_gt09_fraction': .8,
                                     'confidence_mean': .9, 'out_of_bounds_fraction': .01,
                                     'support_used': SETTINGS[policy][1] if slot == 2 else None,
                                     'iterations': SETTINGS[policy][2] if slot == 2 else None}})
    return rows


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    def test_duplicate_replay_requires_same_raw_labels(self):
        path = self.root / 'paired.jsonl'
        rows = records('old')
        self.write(path, rows + [copy.deepcopy(rows[-1])])
        self.assertEqual(len(study.read_records(path, 5, 'old')), 5)
        extra = copy.deepcopy(rows[-1]); extra['label_audit']['tracks_sha256'] = 'c'*64
        self.write(path, rows + [extra])
        with self.assertRaisesRegex(ValueError, 'resumed step'):
            study.read_records(path, 5, 'old')

    def test_missing_or_malformed_coverage_rejected(self):
        path = self.root / 'paired.jsonl'
        rows = records('old')
        variants = [rows[:-1], copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)]
        variants[1][0]['label_audit']['query_time_counts'] = [383]
        variants[2][4]['label_audit']['iterations'] = 6
        variants[3][1]['loss'] = float('nan')
        for variant in variants:
            self.write(path, variant)
            with self.assertRaises(ValueError):
                study.read_records(path, 5, 'old')

    def prepare_arms(self, mutate=None):
        for policy in POLICIES:
            folder = self.root / 'train/seed_0' / policy
            folder.mkdir(parents=True)
            (folder / 'run-config.json').write_text(json.dumps({'policy': policy, 'offline_checkpoint': SETTINGS[policy][0], 'steps': 5}))
            (folder / 'initialization-report.json').write_text(json.dumps({'architecture': 'base', 'initialization_outputs_exact': True, 'added_parameters': 0}))
            (folder / 'cotracker_three_final.pth').write_bytes(b'fixture')
            rows = records(policy)
            if mutate:
                mutate(policy, rows)
            self.write(folder / 'paired-steps.jsonl', rows)

    def audit_mocked_checkpoints(self, cap=15):
        # CPU pairing tests isolate file/record validation, not real checkpoints.
        with patch.object(study, 'validate_checkpoint'), patch.object(study.torch, 'load', side_effect=lambda p, **kw: {'teacher_student_policy': p.parent.name}):
            study.audit(self.root, [0], 5, cap)

    def test_four_arm_audit_passes_and_hashes_files(self):
        self.prepare_arms()
        self.audit_mocked_checkpoints()
        proof = json.loads((self.root / 'training-audit.json').read_text())
        self.assertTrue(proof['complete'])
        self.assertEqual(len(proof['final_checkpoints_sha256']), 4)
        self.assertEqual([p['offline_label_changed_steps_vs_old'] for p in proof['proofs']], [0,1,1,1])
        study.verify_training(self.root, [0], 5)
        (self.root / 'train/seed_0/grid0/cotracker_three_final.pth').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            study.verify_training(self.root, [0], 5)

    def test_pairing_and_unmodified_teachers_are_strict(self):
        def change(policy, rows):
            if policy == 'grid0':
                rows[0]['label_audit']['visibility_sha256'] = 'c'*64
        self.prepare_arms(change)
        with self.assertRaisesRegex(ValueError, 'unmodified teacher'):
            self.audit_mocked_checkpoints()

    def test_runtime_guard_stops_slow_training(self):
        self.prepare_arms()
        with self.assertRaisesRegex(ValueError, 'too slow'):
            self.audit_mocked_checkpoints(cap=.1)

    def test_teacher_sampler_seed_smoke_exercises_all_slots(self):
        from cotracker.utils.teacher_sampling import TeacherPairSampler
        sampler = TeacherPairSampler(3, seed=study.BASE_SEED)
        self.assertEqual([sampler.sample().primary for _ in range(5)], [1,0,1,0,2])

    def test_official_student_report_keeps_splits_seeds_and_negative_d98(self):
        (self.root / 'test-38').mkdir()
        def arm(folder, architecture, count):
            self.assertEqual(architecture, 'base')
            self.assertEqual(count, 38)
            policy = folder.name
            seed = 0 if policy == 'pretrained' else int(folder.parent.name.removeprefix('seed_'))
            delta = {'pretrained': 0., 'old': -.004, 'scaled': -.002, 'grid0': .001, 'iter6': .0008}[policy]
            aggregate = {'DSC': .88 + delta + seed*.0001, 'HD95': 4., 'MASD': 1.6, 'CD': 1.7,
                         'D98': .89 - delta, 'TimeSec': 200.}
            cases = {f'C_{i:03}': {field: aggregate[key] for key, field in study.FIELDS.items()} for i in range(38)}
            return aggregate, cases, 'a'*64
        with patch.object(study, 'verify_training'), patch.object(study, 'load_arm', side_effect=arm):
            study.report(self.root, [0,1,2], 1000, 'test-38')
        def table(name):
            with (self.root / 'test-38' / name).open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))
        summary = {row['Policy']: row for row in table('teacher-student-summary.csv')}
        self.assertEqual(summary['pretrained']['N'], '1')
        self.assertEqual(summary['grid0']['N'], '3')
        self.assertGreater(float(summary['grid0']['DSC_SD']), 0)
        paired = table('teacher-student-paired-deltas.csv')
        gain = next(row for row in paired if row['Seed'] == '0' and row['Candidate'] == 'grid0' and row['Reference'] == 'pretrained')
        self.assertAlmostEqual(float(gain['Delta_DSC']), .001)
        self.assertAlmostEqual(float(gain['Delta_D98']), -.001)
        cases = table('teacher-student-paired-cases.csv')
        self.assertEqual(len(cases), len(paired)*38)
        proof = json.loads((self.root / 'test-38/teacher-student-audit.json').read_text())
        self.assertEqual(proof['split'], 'test-38')
        self.assertEqual(len(proof['metrics_sha256']), 13)


if __name__ == '__main__':
    unittest.main()
