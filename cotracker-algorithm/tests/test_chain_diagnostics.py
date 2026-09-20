import sys
from pathlib import Path
import tempfile
import unittest
import importlib.util
import json

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chain_diagnostics import ChainTrace, compare_traces, mask_difference, overview_rows, vector_difference


class ChainDiagnosticsTests(unittest.TestCase):
    def trace(self, folder, offset=0., start=0, initial=0.):
        trace = ChainTrace(2)
        queries = np.array([[[0., initial, 0.], [0., 4., 4.]]], dtype=np.float32)
        trace.tensor('initial_queries', queries)
        key = trace.begin_level(start, 2, queries)
        trajectory = np.full((1, 3, 2, 2), .2, dtype=np.float32)
        trajectory[:, 1:] += offset
        for name in ('local_trajectory', 'global_trajectory', 'fused_trajectory'):
            trace.tensor(key + '/' + name, trajectory)
        trace.tensor(key + '/local_fusion_weight', np.ones((1, 3, 2)) * .5)
        for name in ('final_trajectory', 'integer_contour'):
            trace.tensor(name, trajectory if name == 'final_trajectory' else trajectory.astype(np.int32))
        for name in ('model_mask', 'native_mask'):
            trace.mask(name, np.zeros((1, 3, 4, 4), dtype=bool))
        trace.observer(key)(0, 'input_track', torch.ones(1, 1, 2, 4) + offset)
        folder.mkdir()
        trace.save(folder)
        return trace

    def test_subpixel_difference_can_leave_integer_and_masks_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.trace(root / 'ref')
            self.trace(root / 'cand', .1)
            report = compare_traces(root / 'ref', root / 'cand')
            rows = {r['stage']: r for r in report['rows']}
            self.assertGreater(rows['final_trajectory']['mean'], 0)
            self.assertEqual(rows['integer_contour']['changed_points'], 0)
            self.assertEqual(rows['native_mask']['changed_pixels'], 0)
            self.assertFalse(report['all_saved_arrays_exact'])

    def test_repeat_exact_and_query_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.trace(root / 'ref')
            self.trace(root / 'same')
            self.assertTrue(compare_traces(root / 'ref', root / 'same')['all_saved_arrays_exact'])
            self.trace(root / 'bad', initial=1.)
            with self.assertRaisesRegex(ValueError, 'Initial query'):
                compare_traces(root / 'ref', root / 'bad')

    def test_boundary_crossing_and_segment_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.trace(root / 'ref')
            self.trace(root / 'cand', .9, start=1)
            report = compare_traces(root / 'ref', root / 'cand')
            self.assertEqual(report['matched_segments'], 0)
            self.assertEqual(len(report['unmatched_reference']), 1)
            contour = next(r for r in report['rows'] if r['stage'] == 'integer_contour')
            self.assertEqual(contour['changed_fraction'], 1.)

    def test_mask_excludes_initial_frame_and_counts_union(self):
        ref = np.zeros((1, 2, 2, 2), dtype=bool)
        cand = ref.copy()
        cand[:, 0] = True
        self.assertEqual(mask_difference(ref, cand)['changed_pixels'], 0)
        cand[0, 1, 0, 0] = True
        result = mask_difference(ref, cand)
        self.assertEqual(result['changed_pixels'], 1)
        self.assertEqual(result['changed_fraction_of_union'], 1.)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            vector_difference(np.zeros((1, 2)), np.zeros((2, 2)))

    def test_feature_sampling_preserves_point_and_channel_axes_without_mutation(self):
        trace = ChainTrace(2)
        source = torch.arange(1 * 49 * 5 * 8).reshape(1, 49, 5, 8).float()
        before = source.clone()
        trace.observer('level')(0, 'input_support', source)
        name = 'level/feature/0/input_support'
        np.testing.assert_array_equal(trace.data[name], source[0, [0, 24, 48]][:, [0, 4]].numpy())
        torch.testing.assert_close(source, before, rtol=0, atol=0)

    def test_overview_is_observation_weighted_not_average_of_means(self):
        rows = [dict(Profile='p', stage='final_trajectory', Case='a', PairChecksPassed=True,
                     count=1, mean=10., p95=10., max=10.),
                dict(Profile='p', stage='final_trajectory', Case='b', PairChecksPassed=True,
                     count=9, mean=0., p95=0., max=0.)]
        result = overview_rows(rows)[0]
        self.assertEqual(result['pooled_mean'], 1.)
        self.assertEqual(result['worst_record_p95_not_pooled'], 10.)
        self.assertNotIn('p95', result)

    def test_checkpoint_corruption_and_end_to_end_summary(self):
        script = Path(__file__).resolve().parents[1] / 'experiments' / 'run_memory_chain_diagnostics.py'
        spec = importlib.util.spec_from_file_location('chain_runner_test', script)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for profile in runner.PROFILES:
                folder = root / 'jobs' / 'A_001' / profile
                folder.parent.mkdir(parents=True, exist_ok=True)
                self.trace(folder, .1 if profile == 'memory_pointwise_fusion' else 0.)
                runner.atomic_json(folder / 'run.json', {'previous_output_matches': None})
                runner.atomic_json(folder / 'complete.json', {'files': {
                    name: runner.file_sha256(folder / name)
                    for name in ('trace.npz', 'trace-meta.json', 'run.json')}})
                self.assertTrue(runner.valid_checkpoint(folder))
            runner.summarize(root, [Path('A_001')], 'test-38')
            summary = json.loads((root / 'summary.json').read_text())
            self.assertTrue(summary['pair_checks_passed'])
            self.assertEqual(summary['prior_output_checks_available'], 0)
            self.assertTrue((root / 'chain-overview.csv').is_file())
            (folder / 'trace.npz').write_bytes(b'corrupted')
            self.assertFalse(runner.valid_checkpoint(folder))


if __name__ == '__main__':
    unittest.main()
