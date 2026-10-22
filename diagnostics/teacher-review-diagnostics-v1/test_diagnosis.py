"""CPU tests exercise exact geometry, rule equivalence, reference gaps and caches."""
import ast
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
os.environ['SIDECAR_CHECKOUT'] = str(REPO)
os.environ['COTRACKER_APP_DIR'] = str(REPO/'cotracker-algorithm')
sys.path.insert(0, str(PACKAGE))
import measures as M
import diagnose as D


def outputs():
    p = np.full((1, 6, 5, 2), 20., np.float32)
    v = np.full((1, 6, 5), .95, np.float32)
    result = [(p.copy(), v.copy(), v.copy()) for _ in range(3)]
    result[1][0][:, :, 0, 0] += 6
    result[2][0][:, :, 0, 0] += 6.1
    result[1][0][:, :, 1, 0] += 1
    result[2][0][:, :, 1, 0] += 1.5
    q = np.concatenate([np.zeros((1, 5, 1), np.float32), p[:, 0]], -1)
    q[0, 3, 0] = 3
    return result, q


class MeasuresTests(unittest.TestCase):
    def test_conflict_agreement_and_query_exclusion(self):
        out, q = outputs()
        r = M.review(out, q, 384, 512, 0)
        self.assertEqual(r['conflict'].sum(), 5)
        self.assertFalse(r['eligible'][:, 0].any())
        self.assertFalse(r['eligible'][:, :4, 3].any())
        np.testing.assert_array_equal(r['weights'][r['conflict']], .25)
        np.testing.assert_array_equal(r['fused'][:, :, 0], out[0][0][:, :, 0])
        self.assertAlmostEqual(r['fused'][0, 1, 1, 0], 20.3125)

    def test_invalid_reviewer_abstains(self):
        out, q = outputs()
        out[1][1][:, 2, 0] = 0
        out[2][0][:, 3, 0] = np.nan
        r = M.review(out, q, 384, 512, 0)
        self.assertEqual(r['conflict'].sum(), 3)
        self.assertTrue(np.isfinite(r['fused']).all())

    def test_threshold_and_cap(self):
        out, q = outputs()
        out[1][0][:, :, 0, 0] = 23
        out[2][0][:, :, 0, 0] = 23.1
        self.assertEqual(M.review(out, q, 384, 512, 0, 2)['conflict'].sum(), 5)
        r = M.review(out, q, 384, 512, 0, 4)
        self.assertEqual(r['conflict'].sum(), 0)
        self.assertLessEqual(np.linalg.norm(r['fused']-r['primary'], axis=-1).max(), .500001)

    def test_known_motion_sign_and_inverse(self):
        t = M.matrices('translation', 60, 384, 512)
        xy = np.array([[50., 30.], [200, 300]], np.float32)
        truth = M.transform_points(xy, t)
        np.testing.assert_allclose(truth[0], xy)
        np.testing.assert_allclose(truth[-1], xy+[12., -8.])
        for motion in D.RULE['synthetic_motions']:
            t = M.matrices(motion, 60, 384, 512)
            target = M.transform_points(xy, t)[-1]
            back = M.transform_points(target, np.linalg.inv(t[-1:]))[0]
            np.testing.assert_allclose(back, xy, atol=5e-5)

    def test_known_truth_and_bounds(self):
        truth = np.array([[[[20., 20.], [-2., 3.]]]], np.float32)
        e, valid = M.known_errors({'a': truth+3}, truth, 40, 40)
        self.assertEqual(valid.sum(), 1)
        self.assertAlmostEqual(e['a'][0, 0, 0], 3*np.sqrt(2), places=5)
        self.assertTrue(np.isnan(e['a'][0, 0, 1]))

    def test_warp_image_matches_forward_point_geometry(self):
        D.load_protocol()
        import torch
        from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset as DS
        y, x = torch.meshgrid(torch.arange(64), torch.arange(80), indexing='ij')
        image = torch.exp(-((x-40.)**2+(y-32.)**2)/4)*255
        source = image[None].repeat(5, 1, 1)
        mask = torch.zeros(1, 64, 80, dtype=torch.bool)
        mask[:, 10:54, 14:66] = True
        with patch.dict(D.RULE, height=64, width=80, frames=5), patch.object(DS, '_read_mha', return_value=source), patch.object(DS, '_read_mask_mha', return_value=mask):
            for motion in D.RULE['synthetic_motions']:
                video, q, truth, transforms, n = D.make_input(Path('.'), dict(Case='A', Mode='synthetic', Motion=motion))
                self.assertEqual(tuple(video.shape), (1, 5, 3, 64, 80))
                np.testing.assert_allclose(truth[:, 0], q.numpy()[..., 1:])
                target = M.transform_points(np.array([[40., 32.]], np.float32), transforms)[-1, 0]
                observed = np.unravel_index(video[0, -1, 0].argmax().item(), (64, 80))[::-1]
                self.assertLess(np.linalg.norm(np.array(observed)-target), 1.)
            real, q, truth, transforms, n = D.make_input(Path('.'), dict(Case='A', Mode='real'))
            self.assertEqual(real.shape[1], 5)
            self.assertIsNone(truth)

    def test_boundary_proxy_does_not_detect_sliding(self):
        mask = np.zeros((1, 20, 20), bool)
        mask[:, 5:15, 5:15] = True
        a = np.array([[[[5., 7.], [5., 10.], [-1., 0.]]]])
        errors = M.boundary_errors({'Primary': a}, mask)['Primary']
        self.assertEqual(errors[0, 0, 0], 0)
        self.assertEqual(errors[0, 0, 1], 0)
        self.assertTrue(np.isnan(errors[0, 0, 2]))

    def test_empty_masks_not_perfect(self):
        e = M.boundary_errors({'p': np.zeros((1, 2, 2, 2))}, np.zeros((2, 20, 20), bool))
        self.assertTrue(np.isnan(e['p']).all())

    def test_common_scoring_and_missing(self):
        e = {k: np.array([1., 3., np.nan]) for k in ('Primary', 'ReviewerA', 'ReviewerB', 'ReviewerMean', 'Fused')}
        e['Fused'][1] = np.nan
        row = M.summarize(np.ones(3, bool), e)
        self.assertEqual((row['Selected'], row['Scored'], row['Unscored']), (3, 1, 2))
        self.assertEqual(row['PrimaryError'], 1)
        row = M.summarize(np.zeros(3, bool), e)
        self.assertIsNone(row['PrimaryError'])

    def test_macro_micro_and_duplicate(self):
        base = dict(Mode='real', Motion='natural', QueryGroup='all', PrimarySlot=0, Threshold=2, Cohort='conflict')
        a = dict(base, Case='A', Selected=1, Scored=1, Unscored=0, PrimaryError=1)
        b = dict(base, Case='B', Selected=9, Scored=9, Unscored=0, PrimaryError=3)
        r = M.aggregate([a, b])[0]
        self.assertEqual(r['Macro_PrimaryError'], 2)
        self.assertEqual(r['Micro_PrimaryError'], 2.8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            M.aggregate([a, a])

    def test_plan_counts_and_no_test_jobs(self):
        cases = {'train-40': [f'A{i}' for i in range(40)], 'validation-10': [f'B{i}' for i in range(10)]}
        self.assertEqual(len(D.job_list(cases, 'full')), 130)
        self.assertEqual(len(D.job_list(cases, 'smoke')), 4)
        self.assertEqual(len({D.key(j) for j in D.job_list(cases, 'full')}), 130)

    def test_exact_original_rule(self):
        p, _ = D.load_protocol()
        out, q = outputs()
        for slot in range(3):
            for threshold in (2, 4):
                D.verify_rule(p, out, q, slot, threshold, M.review(out, q, 384, 512, slot, threshold))
        wrong = M.review(out, q, 384, 512, 0)
        wrong['weights'][:] = 1
        with self.assertRaisesRegex(ValueError, 'differs'):
            D.verify_rule(p, out, q, 0, 2, wrong)

    def test_zero_event_is_missing_not_success(self):
        e = {k: np.array([2.]) for k in ('Primary', 'ReviewerA', 'ReviewerB', 'ReviewerMean', 'Fused')}
        row = dict(Mode='synthetic', Motion='static', QueryGroup='all', PrimarySlot=0,
                   Threshold=2, Cohort='conflict', Case='A', **M.summarize(np.array([False]), e))
        r = M.aggregate([row])[0]
        self.assertEqual(r['CasesScored'], 0)
        self.assertIsNone(r['Macro_ReviewerBetterFraction'])

    def test_rows_and_contrasts(self):
        p, _ = D.load_protocol()
        out, q = outputs()
        masks = np.zeros((6, 384, 512), dtype=bool)
        masks[:, 20:50, 20:50] = True
        rows = D.rows_for(p, dict(Mode='real', Case='A', Motion='natural'), out, q, None, masks)
        self.assertEqual(len(rows), 36)
        summary = M.contrasts(rows)
        self.assertEqual(len(summary), 6)
        one = next(r for r in summary if r['PrimarySlot'] == 0 and r['Threshold'] == 2)
        self.assertEqual(one['ConflictCases'], 1)
        self.assertEqual(one['PairedConflictShuffleCases'], 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            M.contrasts(rows[1:])

    def test_small_contours_do_not_duplicate_queries(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[12:20, 12:20] = True
        points = D.contour(mask, 128)
        self.assertLess(len(points), 128)
        self.assertEqual(len(np.unique(points, axis=0)), len(points))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            D.contour(np.zeros_like(mask), 128)

    def test_cache_mutation_and_safe_paths(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root = Path(tmp)
            D.save(root/'manifest.json', {'test': 1})
            job = dict(Mode='real', Case='A', Motion='natural')
            self.assertIsNone(D.cache(root, job))
            folder = root/'jobs'/D.key(job)
            D.save(folder/'proof.json', {'teacher_parameters_exact': True, 'rules_match_original': True})
            D.save(folder/'rows.json', [])
            np.savez(folder/'outputs.npz', p=np.zeros(1))
            D.save(folder/'verified.json', {'job': job, 'manifest_sha256': D.sha(root/'manifest.json'),
                   'files': {n: D.sha(folder/n) for n in ('outputs.npz', 'rows.json', 'proof.json')}})
            self.assertEqual(D.cache(root, job), folder)
            D.save(folder/'rows.json', [1])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                D.cache(root, job)
            with self.assertRaises(ValueError):
                D.safe(root, '../escaped')
            with self.assertRaises(ValueError):
                D.frozen(root/'manifest.json', {'test': 2})

    def test_no_training_entry(self):
        tree = ast.parse((PACKAGE/'diagnose.py').read_text(encoding='utf-8'))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('backward', calls)
        self.assertNotIn('step', calls)
        ps = (PACKAGE/'run_diagnosis.ps1').read_text(encoding='utf-8')
        self.assertNotIn('target=/datasets/test', ps)
        self.assertNotIn('Bypass', ps)
        self.assertIn("'--pull','never','--network','none'", ps)


if __name__ == '__main__':
    unittest.main()
