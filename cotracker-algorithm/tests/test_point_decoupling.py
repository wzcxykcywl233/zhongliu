"""Exercise real sampling/rasterization with controlled tracking, no weights/GPU."""
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker'), str(ROOT / 'experiments')]
from experiments import POINT_DECOUPLING_EXPERIMENTS as PROFILES, EXPERIMENTS
from point_decoupling import load_cache, save_cache, subset_indices, file_sha
import resources
from replay_point_decoupling import reconstruct
from analyze_point_decoupling import analyze, audit_case
from diagnose_retune_points import FIELDS

spec = importlib.util.spec_from_file_location('point_decoupling_algorithm', ROOT / 'model.py')
algorithm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algorithm)


class PointDecouplingTests(unittest.TestCase):
    def test_controls_and_nested_indices(self):
        self.assertEqual(PROFILES['pd_dense'], EXPERIMENTS['rt_control'])
        self.assertEqual(PROFILES['pd_sparse_native'], EXPERIMENTS['rt_points_0'])
        index = subset_indices(1000, 250)
        self.assertEqual(len(index.unique()), 250)
        self.assertEqual((int(index[0]), int(index[-1])), (0, 999))
        self.assertTrue(torch.all(index[1:] > index[:-1]))
        self.assertEqual(PROFILES['pd_sparse_matched'].query_parent_points, 1000)
        self.assertEqual(PROFILES['pd_dense_thin'].reconstruction_points, 250)

    def test_sampling_tracking_replay_and_corruption_audit(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                frames = np.zeros((40, 32, 3), dtype=np.float32)
                target = np.zeros((40, 32, 1), dtype=np.uint8)
                target[8:31, 7:24] = 1
                captured_queries = {}
                records, predictions = {}, {}

                def fake_tracking(*, video, queries, **kwargs):
                    trajectories = queries[:, None, :, 1:].expand(1, video.shape[1], -1, -1).clone()
                    # Depend on query density so the matched contrast has a real
                    # trajectory difference, not merely a config-name assertion.
                    trajectories[:, 1:, :, 0] += 2.0 if queries.shape[1] == 250 else 1.0
                    captured_queries[current_profile] = queries.clone()
                    values = torch.ones(trajectories.shape[:3])
                    return resources.TrackingResult(trajectories, values, values)

                for current_profile in PROFILES:
                    if current_profile == 'pd_dense_thin':
                        continue
                    cache = root / current_profile / 'point-cache.npz'
                    env = {'COTRACKER_EXPERIMENT': current_profile, 'TRACKRAD_DIAGNOSTICS': '1',
                           'TRACKRAD_POINT_CACHE_PATH': str(cache)}
                    log = io.StringIO()
                    with patch.dict('os.environ', env), patch.object(resources, 'setup_model', return_value=object()), \
                         patch.object(resources, 'hierarchical_forward_pass', side_effect=fake_tracking), redirect_stdout(log):
                        prediction = algorithm.run_algorithm(frames, target, 8., 1.5, 'abdomen')
                    records[current_profile] = json.loads(log.getvalue().split('TRACKRAD_DIAGNOSTICS_JSON=')[1])
                    predictions[current_profile] = prediction
                for historical, profile in [('rt_control', 'pd_dense'), ('rt_points_0', 'pd_sparse_native')]:
                    current_profile = historical
                    with patch.dict('os.environ', {'COTRACKER_EXPERIMENT': historical, 'TRACKRAD_DIAGNOSTICS': '1'}), \
                         patch.object(resources, 'setup_model', return_value=object()), \
                         patch.object(resources, 'hierarchical_forward_pass', side_effect=fake_tracking), redirect_stdout(io.StringIO()):
                        old_prediction = algorithm.run_algorithm(frames, target, 8., 1.5, 'abdomen')
                    np.testing.assert_array_equal(old_prediction, predictions[profile])
                index = subset_indices(1000, 250)
                torch.testing.assert_close(captured_queries['pd_sparse_matched'], captured_queries['pd_dense'][:, index], rtol=0, atol=0)
                self.assertFalse(torch.equal(captured_queries['pd_sparse_matched'], captured_queries['pd_sparse_native']))
                source = root / 'pd_dense' / 'point-cache.npz'
                prediction, record = reconstruct(source, frames, target)
                predictions['pd_dense_thin'], records['pd_dense_thin'] = prediction, record
                self.assertEqual(prediction.shape, frames.shape)
                with self.assertRaisesRegex(ValueError, 'input case'):
                    reconstruct(source, frames + 1, target)
                with patch.object(resources, 'setup_model', side_effect=AssertionError('replay loaded tracker')):
                    repeated, _ = reconstruct(source, frames, target)
                    np.testing.assert_array_equal(prediction, repeated)

                # Commit ten fixture cases using the real query/cache/replay
                # outputs to test stage auditing and official metric reporting.
                for profile in PROFILES:
                    rows = []
                    for i in range(10):
                        case = f'B_{i:03d}'
                        job = root / 'results' / profile / 'checkpoint' / 'jobs' / case
                        output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
                        output.parent.mkdir(parents=True)
                        output.write_bytes(predictions[profile].tobytes())
                        data = json.loads(json.dumps(records[profile]))
                        data['prediction']['output_file_sha256'] = file_sha(output)
                        (job / 'diagnostics.json').write_text(json.dumps(data), encoding='utf-8')
                        (job / 'prediction.json').write_text('{}')
                        (job / '.complete').touch()
                        if profile != 'pd_dense_thin':
                            (job / 'output' / 'point-cache.npz').write_bytes((root / profile / 'point-cache.npz').read_bytes())
                        rows.append({'case_id': case, **{field: .8 for field in FIELDS.values()}})
                    path = root / 'results' / profile / 'metrics.json'
                    path.write_text(json.dumps({'results': rows, 'aggregates': {**{field: .8 for field in FIELDS.values()}, 'total_time': 10.}}))
                with redirect_stdout(io.StringIO()):
                    audit = analyze(root / 'results', 'validation-10')
                self.assertTrue(audit['exact_matched_queries'])
                self.assertTrue((root / 'results' / 'point-decoupling-validation-10-matched-deltas.csv').is_file())
                for profile, historical in [('pd_dense', 'rt_control'), ('pd_sparse_native', 'rt_points_0')]:
                    destination = root / 'prior' / 'final' / 'validation-10' / historical
                    shutil.copytree(root / 'results' / profile, destination)
                    for path in destination.rglob('diagnostics.json'):
                        data = json.loads(path.read_text())
                        data['profile'] = historical
                        path.write_text(json.dumps(data))
                with redirect_stdout(io.StringIO()):
                    audit = analyze(root / 'results', 'validation-10', root / 'prior')
                self.assertEqual(audit['historical_output_checks'], 20)
                dense_job = root / 'results' / 'pd_dense' / 'checkpoint' / 'jobs' / 'B_000'
                (dense_job / 'output' / 'point-cache.npz').write_bytes(b'broken archive')
                with self.assertRaises((ValueError, EOFError)):
                    audit_case(root / 'results', 'B_000')
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
