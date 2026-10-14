import argparse
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import common as c
import predictor as p
import worker as w


class FakeResources:
    def __init__(self):
        self.calls = []

    def forward_pass(self, model, video, queries, **kwargs):
        self.calls.append(('plain', queries.clone(), kwargs))
        return self.result(video, queries)

    def hierarchical_forward_pass(self, model, video, queries, **kwargs):
        self.calls.append(('hierarchical', queries.clone(), kwargs))
        return self.result(video, queries)

    def result(self, video, queries):
        n, t = queries.shape[1], video.shape[1]
        return SimpleNamespace(trajectories=queries[:, None, :, 1:].expand(1, t, n, 2).clone(),
                               visibility=torch.full((1, t, n), .95), confidence=torch.full((1, t, n), .9))


class TeacherTests(unittest.TestCase):
    def test_all_profiles_one_factor_and_order(self):
        spec, profiles, resolved = c.graph()
        self.assertEqual(len(profiles), 32)
        self.assertEqual(spec['query_scope'], 'first-frame-boundary-only')
        self.assertEqual(c.group_profiles('weights'), ['old', 'scaled'])
        for name, row in profiles.items():
            parent = row['parent']
            if parent:
                diff = [k for k in resolved[name] if resolved[name][k] != resolved[parent][k]]
                self.assertEqual(diff, list(row['change']))
            self.assertNotIn('mamba', str(resolved[name]).lower())
        bad = c.read(c.PACKAGE / 'experiments.json')
        bad['profiles'][1]['change']['iterations'] = 6
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            c.graph(bad)
        self.assertEqual(resolved['anchor_rollback']['occlusion_merge'], False)
        self.assertEqual(resolved['latest2']['memory_slots'], 2)
        self.assertEqual(resolved['latest4']['memory_slots'], 4)

    def test_every_profile_api_target_identity(self):
        _, _, configs = c.graph()
        video = torch.ones((1, 4, 1, 12, 16))
        queries = torch.tensor([[[0., 4., 4.], [0., 8., 8.]]])
        original = queries.clone()
        for name, config in configs.items():
            resource = FakeResources()
            extra = queries[:, :1].repeat(1, config['extra_border_points'], 1) if config['extra_border_points'] else None
            arrays, diag = p.track(None, video, queries, config, 16, extra, 'cpu', resource)
            self.assertEqual(arrays[0].shape, (1, 4, 2, 2), name)
            self.assertTrue(torch.equal(arrays[0][:, 0], original[..., 1:]), name)
            self.assertTrue(torch.equal(queries, original), name)
            self.assertEqual(diag['fixed_queries_sha256'], p.tensor_sha(original))
            method, called, kwargs = resource.calls[0]
            self.assertTrue(torch.equal(called[:, :2], original))
            if method == 'hierarchical':
                self.assertEqual(kwargs['query_memory_slots'], config['memory_slots'] if config['memory_mode'] != 'none' else 0)
                self.assertEqual(kwargs['query_state_inheritance'], config['state'])
                self.assertEqual(kwargs['adaptive_anchor'], config['adaptive_anchor'])
            if config['support'] == 'uniform100':
                self.assertEqual(called.shape[1], 102)
            if name == 'extra750':
                self.assertEqual(called.shape[1], 752)

    def test_nonzero_time_rejected_before_hierarchy(self):
        resource = FakeResources()
        config = c.graph()[2]['hierarchy']
        with self.assertRaisesRegex(ValueError, 'first-frame'):
            p.track(None, torch.ones(1, 4, 1, 12, 16), torch.tensor([[[2., 3., 4.]]]), config, 0,
                    device='cpu', resources_module=resource)
        self.assertEqual(resource.calls, [])

    def test_context_seed_is_fixed_and_frame_bounds(self):
        v = torch.zeros(1, 3, 1, 12, 16)
        a, b = p.uniform_support(v, 4), p.uniform_support(v, 4)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(bool((a[..., 0] < 3).all()))
        self.assertTrue(bool((a[..., 1] <= 15).all()))
        self.assertTrue(bool((a[..., 2] <= 11).all()))

    def test_axis_and_normalization(self):
        raw = np.arange(7 * 5 * 4).reshape(7, 5, 4)
        canonical = p.canonical(raw)
        self.assertEqual(canonical.shape, (4, 5, 7))
        self.assertEqual(canonical[3, 2, 6], raw[6, 2, 3])
        self.assertTrue(np.array_equal(p.preprocess(canonical, 'raw', (5, 7))[0, :, 0], canonical))
        self.assertEqual(p.preprocess(canonical, 'raw').shape, (1, 4, 1, 384, 512))
        normalized = p.preprocess(canonical, 'robust_1_99', (5, 7))
        self.assertEqual(normalized.shape, (1, 4, 1, 5, 7))
        self.assertEqual(float(normalized.min()), 0)
        self.assertEqual(float(normalized.max()), 255)

    def test_hash_cache_tamper_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            c.save(folder / 'payload.json', {'a': 1})
            self.assertFalse(c.verify_job(folder, 't'))
            c.commit_job(folder, 't', ['payload.json'])
            self.assertTrue(c.verify_job(folder, 't'))
            with self.assertRaisesRegex(ValueError, 'another run'):
                c.verify_job(folder, 'other')
            c.save(folder / 'payload.json', {'a': 2})
            with self.assertRaisesRegex(ValueError, 'changed'):
                c.verify_job(folder, 't')
            case = folder / 'C_001'
            c.save(case / 'frame-rate.json', 4)
            c.save(case / 'b-field-strength.json', .35)
            c.save(case / 'scanned-region.json', 'thorax')
            job = w.prediction('C_001', folder, '2026-10-14T00:00:00+00:00', '2026-10-14T00:00:02+00:00')
            self.assertEqual([x['value'] for x in job['inputs'][:3]], [4, .35, 'thorax'])
            self.assertEqual(job['pk'], 'jobs/C_001')
            c.save(case / 'frame-rate.json', True)
            with self.assertRaises(ValueError):
                w.prediction('C_001', folder, '', '')

    def test_split_disjointness(self):
        rows = []
        for split, count in c.SPLITS.items():
            for i in range(count):
                case = split.replace('-', '_') + '_' + str(i)
                rows.append({'Split': split, 'Case': case, 'File': f'C:\\data\\{split}\\{case}\\images\\a.mha', 'SHA256': 'abc'})
        parsed = c.protocol_files({'Dataset': rows})
        self.assertEqual([len(x) for x in parsed.values()], [40, 10, 38])
        rows[-1]['Case'] = rows[0]['Case']
        rows[-1]['File'] = rows[0]['File']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            c.protocol_files({'Dataset': rows})

    def test_official_metrics_not_recomputed_and_missing_cases_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metrics.json'
            data = {'aggregates': {v: .8 for v in c.METRICS.values()},
                    'results': [{'case_id': 'A_001', **{v: .7 for v in c.METRICS.values()}}]}
            data['aggregates']['total_time'] = 12
            c.save(path, data)
            self.assertEqual(c.official(path, ['A_001'])[0]['DSC'], .8)
            with self.assertRaises(ValueError):
                c.official(path, ['A_001', 'B_001'])

    def test_selection_uses_validation_only_and_parents(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            c.save(output / 'manifest.json', {'test': 1})
            args = argparse.Namespace(output=output, group='memory')
            names = c.group_profiles('memory')
            for name in names:
                c.save(output / 'validation-10' / name / 'evaluation/metrics.json', {'name': name})
            def fake(args, split, name):
                self.assertEqual(split, 'validation-10')
                score = .8 + names.index(name) * .001
                return ({'DSC': score, 'HD95': 4., 'MASD': 2., 'CD': 2., 'D98': .9}, {})
            with patch.object(w, 'metrics_for', fake):
                value = w.selection_value(args)
                self.assertEqual(value['winners'][0], 'diverse')
                self.assertIn('topk', value['test_profiles'])
                self.assertIn('old', value['test_profiles'])
                self.assertLessEqual(len(value['test_profiles']), 8)
                w.select(args)
                w.verify_selection(args)
                args.group = 'weights'
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    w.select(args)

    def test_real_query_and_mask_mha_roundtrip(self):
        # Real image libraries, non-square geometry, no network/model/GPU.
        p.activate()
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp)
            folder = dataset / 'A_001'
            (folder / 'images').mkdir(parents=True)
            (folder / 'targets').mkdir()
            masks = np.zeros((4, 75, 103), np.uint8)
            masks[:, 15:55, 25:80] = 1
            for leaf, array in [('images/A_001_frames.mha', masks * 100), ('targets/A_001_first_label.mha', masks[:1])]:
                sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0))), str(folder / leaf))
            image, array, first = w.input_case(dataset, 'A_001')
            self.assertEqual(array.shape, masks.shape)
            queries = p.query_points(first, 250)
            outputs = FakeResources().result(torch.zeros(1, 4, 1, 384, 512), queries)
            mask = p.render([outputs.trajectories, outputs.visibility, outputs.confidence], (75, 103), first)
            self.assertTrue(np.array_equal(mask[0], first))
            self.assertEqual(mask.shape, masks.shape)
            self.assertGreater(2 * (mask[1] & masks[1]).sum() / (mask[1].sum() + masks[1].sum()), .95)
            out = sitk.GetImageFromArray(mask.transpose(2, 1, 0).astype(np.uint8))
            out.CopyInformation(image)
            sitk.WriteImage(out, str(dataset / 'output.mha'), True)
            self.assertTrue(np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(dataset / 'output.mha'))), mask.transpose(2, 1, 0)))

    def test_frozen_plan_restarts_and_changed_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(protocol=root / 'protocol', study=root / 'study', weights=root / 'weights',
                                      output=root / 'output', inference_image='sha256:' + '1' * 64,
                                      evaluation_image='sha256:' + '2' * 64,
                                      datasets={s: root / s for s in c.SPLITS})
            args.output.mkdir()
            rows = []
            for split, count in c.SPLITS.items():
                for i in range(count):
                    case = split.replace('-', '_') + '_' + str(i)
                    file = args.datasets[split] / case / 'images/a.mha'
                    file.parent.mkdir(parents=True)
                    file.write_bytes(b'fixture')
                    rows.append({'Split': split, 'Case': case,
                                 'File': f'C:\\data\\{split}\\{case}\\images\\a.mha', 'SHA256': c.sha(file)})
            numerical = root / 'app'
            sources = []
            for relative in ('resources/model.py', 'ext/co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py'):
                file = numerical / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('# fake numerical source', encoding='utf-8')
                sources.append({'File': 'cotracker-algorithm/' + relative, 'SHA256': c.sha(file)})
            # The frozen host source may list a new test absent from reuse images.
            sources.append({'File': 'cotracker-algorithm/tests/not_in_runtime.py', 'SHA256': 'fake'})
            args.weights.mkdir()
            weights = []
            for name in ('baseline_offline.pth', 'scaled_offline.pth'):
                file = args.weights / name
                file.write_bytes(b'fake weight ' + name.encode())
                weights.append({'Name': name, 'SHA256': c.sha(file)})
            c.save(args.protocol / 'frozen-run.json', {'Dataset': rows, 'Weights': weights})
            c.save(args.study / 'frozen-study.json', {'Source': sources,
                   'OldProtocolSHA256': c.sha(args.protocol / 'frozen-run.json'),
                   'Images': {'Evaluation': args.evaluation_image}})
            with patch.object(w, 'NUMERICAL_ROOT', numerical):
                w.plan(args)
                manifest_hash = c.sha(args.output / 'manifest.json')
                w.plan(args)
                self.assertEqual(c.sha(args.output / 'manifest.json'), manifest_hash)
                w.verify(args, 'validation-10', 'scaled')
                file = next(args.datasets['validation-10'].rglob('*.mha'))
                file.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Dataset changed'):
                    w.verify(args, 'validation-10', 'scaled')
                with self.assertRaisesRegex(ValueError, 'Dataset changed'):
                    w.plan(args)

    def test_test38_report_has_38_official_cases_and_matched_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(output=root, split='test-38', group='weights')
            spec, _, configs = c.graph()
            case_ids = ['X_' + str(i).zfill(3) for i in range(38)]
            manifest = {'inputs': {'test-38/' + case + '/a.mha': 'fake' for case in case_ids},
                        'resolved': configs, 'spec': spec}
            c.save(root / 'manifest.json', manifest)
            c.save(root / 'selection.json', {'test_profiles': ['old', 'scaled']})
            for name in ('old', 'scaled'):
                for case in case_ids:
                    folder = root / 'test-38' / name / 'jobs' / case
                    c.save(folder / 'diagnostics.json', {'labels': {'visibility_mean': .95}, 'seconds_tracking': 1})
                    c.save(folder / 'prediction.json', {'case': case})
                    (folder / c.MASK).parent.mkdir(parents=True)
                    (folder / c.MASK).write_bytes(b'fake mha for cache test')
                    np.savez(folder / 'tracks.npz', queries=np.zeros((1, 2, 3)),
                             trajectories=np.zeros((1, 3, 2, 2)),
                             visibility=np.full((1, 3, 2), .95), confidence=np.full((1, 3, 2), .9))
                    c.commit_job(folder, w.token(args, 'test-38', name), [c.MASK, 'tracks.npz', 'diagnostics.json', 'prediction.json'])
                values = {v: (.8 if name == 'old' else .81) for v in c.METRICS.values()}
                data = {'results': [{'case_id': case, **values} for case in case_ids],
                        'aggregates': {**values, 'total_time': 38}}
                evaluation = root / 'test-38' / name / 'evaluation'
                c.save(evaluation / 'metrics.json', data)
                c.commit_job(evaluation, w.evaluation_token(args, 'test-38', name), ['metrics.json'])
            with patch.object(w, 'verify_selection'):
                w.report(args)
                summary = c.read(root / 'teacher-test-38-summary.json')
                self.assertEqual(summary['cases'], 38)
                self.assertEqual(summary['split'], 'test-38')
                self.assertFalse(summary['training'])
                with (root / 'teacher-test-38-results.csv').open(encoding='utf-8-sig') as f:
                    import csv
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[1]['DSC'], '0.81')
                self.assertTrue(all(r['ReportKind'] == 'teacher-label-generator' for r in rows))
                target = root / 'test-38/scaled/jobs' / case_ids[0] / 'tracks.npz'
                np.savez(target, queries=np.ones((1, 2, 3)))
                with self.assertRaisesRegex(ValueError, 'changed'):
                    w.report(args)

    def test_actual_official_evaluator_accepts_generated_prediction(self):
        # Use the repository evaluator itself on a small non-square MHA fixture.
        import SimpleITK as sitk
        evaluation_source = c.PACKAGE.parents[1] / 'evaluation'
        sys.path.insert(0, str(evaluation_source))
        import evaluate as evaluator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = 'A_001'
            masks = np.zeros((4, 40, 50), np.uint8)
            masks[:, 10:30, 10:35] = 1
            truth = root / 'dataset' / case / 'targets'
            truth.mkdir(parents=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(masks.transpose(2, 1, 0))), str(truth / (case + '_labels.mha')))
            output = root / 'input/jobs' / case / c.MASK
            output.parent.mkdir(parents=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(masks.transpose(2, 1, 0))), str(output), True)
            for leaf, value in [('frame-rate.json', 4), ('b-field-strength.json', 1.5), ('scanned-region.json', 'abdomen')]:
                c.save(root / 'dataset' / case / leaf, value)
            job = w.prediction(case, root / 'dataset', '2026-10-14T00:00:00+00:00', '2026-10-14T00:00:02+00:00')
            with patch.object(evaluator, 'INPUT_DIRECTORY', root / 'input'), patch.object(evaluator, 'GROUND_TRUTH_DIRECTORY', root / 'dataset'):
                result = evaluator.process(job)
            self.assertEqual(result['case_id'], case)
            self.assertEqual(result['dice_similarity_coefficient'], 1)
            self.assertAlmostEqual(result['relative_d98_dose'], 1)
            self.assertEqual(result['total_runtime'], 2)

    def test_inference_commit_and_resume_without_reading_future_labels(self):
        import SimpleITK as sitk
        p.activate()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = 'A_001'
            dataset = root / 'dataset'
            folder = dataset / case
            (folder / 'images').mkdir(parents=True)
            (folder / 'targets').mkdir()
            masks = np.zeros((4, 75, 103), np.uint8)
            masks[:, 15:55, 25:80] = 1
            for leaf, array in [('images/A_001_frames.mha', masks * 100), ('targets/A_001_first_label.mha', masks[:1])]:
                sitk.WriteImage(sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0))), str(folder / leaf))
            # There is deliberately NO labels.mha in this inference fixture.
            for leaf, value in [('frame-rate.json', 4), ('b-field-strength.json', .35), ('scanned-region.json', 'abdomen')]:
                c.save(folder / leaf, value)
            args = argparse.Namespace(output=root / 'output', weights=root / 'weights', split='validation-10',
                                      profile='old', datasets={'validation-10': dataset})
            args.weights.mkdir()
            checkpoint = args.weights / 'baseline_offline.pth'
            checkpoint.write_bytes(b'fake frozen weight')
            spec, _, configs = c.graph()
            manifest = {'inputs': {'validation-10/' + k: v for k, v in c.inventory(dataset).items()},
                        'resolved': configs, 'weights': {'baseline_offline.pth': c.sha(checkpoint)}, 'spec': spec}
            c.save(args.output / 'manifest.json', manifest)
            c.save(args.output / 'smoke/summary.json', {'mock': True})
            c.commit_job(args.output / 'smoke', c.sha(args.output / 'manifest.json'), ['summary.json'])
            def mocked_track(model, video, queries, config, seed, extra):
                self.assertEqual(video.shape[-2:], (384, 512))
                r = FakeResources().result(video, queries)
                return [r.trajectories, r.visibility, r.confidence], {'mocked': True}
            with patch.object(w, 'verify', return_value=manifest), patch.object(p, 'load_teacher', return_value=object()) as loader, \
                    patch.object(p, 'track', side_effect=mocked_track) as tracker, patch.object(torch.cuda, 'synchronize'):
                w.infer(args)
                w.infer(args)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(tracker.call_count, 1)
            final = args.output / 'validation-10/old/jobs' / case
            self.assertTrue(c.verify_job(final, w.token(args, 'validation-10', 'old')))
            pred = p.canonical(sitk.GetArrayFromImage(sitk.ReadImage(str(final / c.MASK))))
            self.assertEqual(pred.shape, masks.shape)
            self.assertTrue(np.array_equal(pred[0], masks[0]))
            index = c.read(args.output / 'validation-10/old/predictions.json')
            self.assertEqual(index[0]['pk'], 'jobs/A_001')


if __name__ == '__main__':
    unittest.main()
