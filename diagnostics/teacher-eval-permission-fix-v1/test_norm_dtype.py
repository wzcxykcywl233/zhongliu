import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import norm_dtype as n

ORIGINAL = Path(__file__).resolve().parent.parent / 'teacher-single-factor-v1'
c, w = n.original_modules(ORIGINAL)
import predictor as p


class NormDtypeTests(unittest.TestCase):
    def fixture(self, root, images=False):
        output, dataset, weights = [root / k for k in ('output', 'dataset', 'weights')]
        weights.mkdir()
        checkpoint = weights / 'scaled_offline.pth'
        checkpoint.write_bytes(b'fake frozen teacher')
        case_ids = [f'A_{i:03d}' for i in range(10)]
        masks = np.zeros((4, 20, 24), np.uint8)
        masks[:, 5:15, 6:18] = 1
        if images:
            import SimpleITK as sitk
            for case in case_ids:
                folder = dataset / case
                (folder / 'images').mkdir(parents=True)
                (folder / 'targets').mkdir()
                # No future-frame labels exist: the inference must not read them.
                for leaf, data in [(f'images/{case}_frames.mha', masks * 100),
                                   (f'targets/{case}_first_label.mha', masks[:1])]:
                    sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0))), str(folder / leaf))
                for leaf, value in [('frame-rate.json', 4), ('b-field-strength.json', .35), ('scanned-region.json', 'abdomen')]:
                    c.save(folder / leaf, value)
            inputs = {'validation-10/' + k: v for k, v in c.inventory(dataset).items()}
        else:
            inputs = {'validation-10/' + case + '/image.mha': 'fixture' for case in case_ids}
        spec, _, configs = c.graph()
        manifest = {'package': c.package_hashes(), 'inputs': inputs, 'resolved': configs, 'spec': spec,
                    'weights': {'scaled_offline.pth': c.sha(checkpoint)}}
        c.save(output / 'manifest.json', manifest)
        c.save(output / 'smoke/summary.json', {'mock': True})
        c.commit_job(output / 'smoke', c.sha(output / 'manifest.json'), ['summary.json'])
        args = argparse.Namespace(output=output, weights=weights, split='validation-10', profile='norm',
                                  datasets={'validation-10': dataset})
        return args, manifest, case_ids, masks

    def arguments(self, args, action='infer', profile='norm', group='core', split='validation-10'):
        return ['--action', action, '--output', str(args.output), '--profile', profile, '--group', group, '--split', split]

    def test_numpy2_regression_and_float32_convolution(self):
        array = np.arange(120, dtype=np.float32).reshape(3, 4, 10)
        original = p.preprocess(array, 'robust_1_99', (4, 10))
        conv = torch.nn.Conv2d(1, 2, 1)
        if int(np.__version__.split('.')[0]) >= 2:
            self.assertEqual(original.dtype, torch.float64)
            with self.assertRaisesRegex(RuntimeError, 'double'):
                conv(original[0])
        evidence = {}
        corrected = n.corrected_preprocess(array, 'robust_1_99', (4, 10), p.preprocess, evidence)
        self.assertEqual(corrected.dtype, torch.float32)
        self.assertEqual(conv(corrected[0]).shape, (3, 2, 4, 10))
        self.assertEqual(evidence['resize_input_dtype'], 'float32')
        self.assertEqual(evidence['output_tensor_dtype'], 'torch.float32')
        self.assertEqual(float(corrected.min()), 0)
        self.assertEqual(float(corrected.max()), 255)

    def test_same_formula_cast_before_same_resize_and_raw_exact(self):
        array = np.random.default_rng(42).normal(size=(3, 7, 9)).astype(np.float32)
        low, high = np.percentile(array, (1, 99))
        expected = p.preprocess((np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.float32), 'raw', (11, 13))
        corrected = n.corrected_preprocess(array, 'robust_1_99', (11, 13), p.preprocess)
        self.assertTrue(torch.equal(corrected, expected))
        self.assertTrue(torch.equal(p.preprocess(array, 'raw', (11, 13)),
                                   n.corrected_preprocess(array, 'raw', (11, 13), p.preprocess)))
        with self.assertRaisesRegex(ValueError, 'Unknown intensity'):
            n.corrected_preprocess(array, 'invalid', (11, 13), p.preprocess)

    def test_flat_and_nonfinite_input_stop_before_model(self):
        for array in [np.ones((2, 3, 4)), np.full((2, 3, 4), np.nan)]:
            with self.assertRaisesRegex(ValueError, 'Non-identifiable'):
                n.corrected_preprocess(array, 'robust_1_99', (3, 4), p.preprocess)

    def test_proof_freezes_original_package_and_rejects_changed_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            args, manifest, _, _ = self.fixture(Path(temp))
            watched = c.sha(args.output / 'manifest.json')
            _, proof_path, proof = n.prepare_norm(c, args.output, 'validation-10')
            self.assertEqual(c.read(proof_path), proof)
            self.assertEqual(c.sha(args.output / 'manifest.json'), watched)
            self.assertEqual(c.package_hashes(), manifest['package'])
            proof['wrapper_sha256'] = 'different'
            c.save(proof_path, proof)
            with self.assertRaisesRegex(ValueError, 'proof missing or changed'):
                n.prepare_norm(c, args.output, 'validation-10')

    def test_no_adoption_or_overwrite_of_legacy_norm_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, cases, _ = self.fixture(Path(temp))
            marker = args.output / 'validation-10/norm/jobs' / cases[0] / 'verified.json'
            c.save(marker, {'legacy': True})
            watched = c.sha(marker)
            with self.assertRaisesRegex(ValueError, 'no dtype provenance'):
                n.prepare_norm(c, args.output, 'validation-10')
            self.assertEqual(c.sha(marker), watched)
            self.assertFalse((args.output / 'dtype-repair/validation-10/norm.json').exists())

    def test_non_norm_actions_delegate_without_mutating_preprocess_or_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = self.fixture(Path(temp))
            original = p.preprocess
            with patch.object(w, 'main', return_value=17) as main:
                self.assertEqual(n.run(self.arguments(args, profile='scaled', group='weights'), ORIGINAL), 17)
                self.assertIs(p.preprocess, original)
                main.assert_called_once()
            self.assertFalse((args.output / 'dtype-repair').exists())

    def test_report_select_and_test_cannot_skip_norm_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, _ = self.fixture(Path(temp))
            with patch.object(w, 'main') as main:
                for action, split in [('report', 'validation-10'), ('select', 'validation-10'), ('infer', 'test-38')]:
                    with self.assertRaisesRegex(ValueError, 'proof missing or changed'):
                        n.run(self.arguments(args, action=action, split=split), ORIGINAL)
                main.assert_not_called()

    def run_inference_fixture(self, interrupt=False):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as temp:
            args, manifest, cases, masks = self.fixture(Path(temp), images=True)
            raw_payload = args.output / 'validation-10/scaled/jobs/existing/control.txt'
            raw_payload.parent.mkdir(parents=True)
            raw_payload.write_bytes(b'completed raw control must remain untouched')
            watched = {args.output / 'manifest.json': c.sha(args.output / 'manifest.json'), raw_payload: c.sha(raw_payload)}
            package_hashes = c.package_hashes()
            original_preprocess, original_commit = p.preprocess, w.commit_job
            queries = torch.tensor([[[0., 100., 100.], [0., 200., 200.]]])
            convolution = torch.nn.Conv2d(1, 1, 1)
            calls = []

            def tracking(model, video, query, config, seed, extra):
                self.assertEqual(video.dtype, torch.float32)
                convolution(video[0])  # Real dtype-sensitive convolution; no GPU/model weights.
                calls.append(seed)
                if interrupt and len(calls) == 3:
                    raise RuntimeError('simulated interruption')
                count, frames = query.shape[1], video.shape[1]
                return [query[:, None, :, 1:].expand(1, frames, count, 2).clone(),
                        torch.full((1, frames, count), .95), torch.full((1, frames, count), .9)], {'mocked': True}

            with patch.object(w, 'main', side_effect=lambda: w.infer(args)), \
                    patch.object(w, 'verify', return_value=manifest), \
                    patch.object(p, 'load_teacher', return_value=object()) as loader, \
                    patch.object(p, 'query_points', return_value=queries), \
                    patch.object(p, 'track', side_effect=tracking), \
                    patch.object(p, 'render', return_value=masks.astype(bool)), \
                    patch.object(torch.cuda, 'synchronize'), contextlib.redirect_stdout(io.StringIO()):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        n.run(self.arguments(args), ORIGINAL)
                    first = args.output / 'validation-10/norm/jobs' / cases[0]
                    committed_before = {p: c.sha(p) for p in first.rglob('*') if p.is_file()}
                    self.assertIs(p.preprocess, original_preprocess)
                    self.assertIs(w.commit_job, original_commit)
                n.run(self.arguments(args), ORIGINAL)
                tracked = len(calls)
                loaded = loader.call_count
                all_outputs = {path: c.sha(path) for path in (args.output / 'validation-10/norm/jobs').rglob('*') if path.is_file()}
                n.run(self.arguments(args), ORIGINAL)
                self.assertEqual(len(calls), tracked)
                self.assertEqual(loader.call_count, loaded)
                self.assertEqual({path: c.sha(path) for path in all_outputs}, all_outputs)
                if interrupt:
                    self.assertEqual({path: c.sha(path) for path in committed_before}, committed_before)
            self.assertEqual({path: c.sha(path) for path in watched}, watched)
            self.assertEqual(c.package_hashes(), package_hashes)
            self.assertIs(p.preprocess, original_preprocess)
            self.assertIs(w.commit_job, original_commit)
            n.validate_norm_jobs(c, args.output, 'validation-10', True)
            self.assertEqual(len(list((args.output / 'validation-10/norm/jobs').glob('*/verified.json'))), 10)
            for case in cases:
                final = args.output / 'validation-10/norm/jobs' / case
                diagnostic = c.read(final / 'diagnostics.json')
                self.assertEqual(diagnostic[n.DIAGNOSTIC_KEY]['output_tensor_dtype'], 'torch.float32')
                result = p.canonical(sitk.GetArrayFromImage(sitk.ReadImage(str(final / c.MASK))))
                self.assertTrue(np.array_equal(result, masks))
                self.assertTrue(c.verify_job(final, w.token(args, 'validation-10', 'norm'), w.JOB_OUTPUTS))
            # Even a newly hash-committed diagnostic without the dtype evidence must be rejected.
            final = args.output / 'validation-10/norm/jobs' / cases[0]
            diagnostic = c.read(final / 'diagnostics.json')
            del diagnostic[n.DIAGNOSTIC_KEY]
            c.save(final / 'diagnostics.json', diagnostic)
            c.commit_job(final, w.token(args, 'validation-10', 'norm'), w.JOB_OUTPUTS)
            with self.assertRaisesRegex(ValueError, 'float32 provenance'):
                n.validate_norm_jobs(c, args.output, 'validation-10', True)

    def test_actual_worker_mha_commit_float32_and_exact_resume(self):
        self.run_inference_fixture()

    def test_interrupted_worker_preserves_completed_norm_cases_and_resumes(self):
        self.run_inference_fixture(interrupt=True)


if __name__ == '__main__':
    unittest.main()
