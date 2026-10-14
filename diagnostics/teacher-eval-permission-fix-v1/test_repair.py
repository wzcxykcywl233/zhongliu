import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import repair as r


class PermissionRepairTests(unittest.TestCase):
    def fixture(self, root):
        original, study, results, evaluator = [root / x for x in ('original', 'study', 'results', 'evaluator')]
        original.mkdir()
        (original / 'worker.py').write_bytes(b'original worker')
        (original / 'run_teacher_experiments.ps1').write_bytes(b'original launcher')
        evaluator.mkdir()
        for leaf in r.EVALUATOR_FILES:
            (evaluator / leaf).write_text('# unchanged evaluator fixture', encoding='utf-8')
        source = {'Images': {'Evaluation': 'sha256:' + '1' * 64},
                  'Source': [{'File': 'evaluation/' + leaf, 'SHA256': r.sha(evaluator / leaf)} for leaf in r.EVALUATOR_FILES]}
        r.save(study / 'frozen-study.json', source)
        cases = ['A_' + str(i).zfill(3) for i in range(10)]
        manifest = {'package': r.package_hashes(original), 'study_sha256': r.sha(study / 'frozen-study.json'),
                    'resolved': {'old': {}}, 'inputs': {'validation-10/' + case + '/image.mha': 'placeholder' for case in cases},
                    'inference_image': 'sha256:' + '2' * 64, 'evaluation_image': source['Images']['Evaluation']}
        r.save(results / 'manifest.json', manifest)
        token = hashlib.sha256((r.sha(results / 'manifest.json') + '/validation-10/old').encode()).hexdigest()
        profile = results / 'validation-10/old'
        (profile / 'evaluation').mkdir(parents=True)
        r.save(profile / 'predictions.json', [{'pk': 'jobs/' + case} for case in cases])
        for case in cases:
            job = profile / 'jobs' / case
            for leaf in r.JOB_OUTPUTS:
                file = job / leaf
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'original prediction payload')
            r.save(job / 'verified.json', {'complete': True, 'token': token,
                                         'outputs': {leaf: r.sha(job / leaf) for leaf in r.JOB_OUTPUTS}})
        self.modules = {}
        for name in r.DEPENDENCIES:
            file = evaluator / (name + '_fixture.py')
            file.write_text('# immutable dependency', encoding='utf-8')
            self.modules[name] = SimpleNamespace(__file__=str(file), __version__='fixture-1')
        self.evaluator_user = {'uid': 999, 'gid': 998}
        runtime_path = root / 'runtime.json'
        with patch.object(r, 'identity', return_value=self.evaluator_user), patch.object(r.importlib, 'import_module', side_effect=lambda n: self.modules[n]), contextlib.redirect_stdout(io.StringIO()):
            audit = r.runtime(evaluator)
        r.save(runtime_path, audit)
        self.runtime_path = runtime_path
        self.cases = cases
        return original, study, results, evaluator, profile

    def prepare(self, original, study, results):
        return r.prepare(results, original, study, 'validation-10', 'old', self.runtime_path)

    def response(self, receipt, payload=None):
        proof = r.read(receipt)
        values = {key: .8 for key in ('dice_similarity_coefficient', 'hausdorff_distance_95',
                                    'surface_distance_average', 'center_distance', 'relative_d98_dose')}
        if payload is None:
            payload = r.json.dumps({'results': [{'case_id': c, **values} for c in self.cases],
                                    'aggregates': {**values, 'total_time': 12}}, indent=4).encode()
        return {'schema': 2, 'type': 'teacher-official-metrics', 'preflight_sha256': r.sha(receipt),
                'manifest_sha256': proof['manifest_sha256'], 'evaluation_image': proof['evaluation_image'],
                'identity': self.evaluator_user, 'metrics_json': payload.decode(),
                'metrics_sha256': hashlib.sha256(payload).hexdigest()}

    def test_manifest_jobs_original_package_preserved_and_previous_metrics_snapshotted(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            watched = {p: r.sha(p) for base in (original, profile / 'jobs') for p in base.rglob('*') if p.is_file()}
            watched[results / 'manifest.json'] = r.sha(results / 'manifest.json')
            metrics = profile / 'evaluation/metrics.json'
            metrics.write_bytes(b'{truncated')
            receipt = self.prepare(original, study, results)
            proof = r.read(receipt)
            self.assertEqual(len(proof['cases']), 10)
            self.assertEqual(proof['owner_uid'], (profile / 'evaluation').stat().st_uid)
            self.assertEqual(proof['original_evaluator_runtime']['identity'], self.evaluator_user)
            self.assertEqual(metrics.read_bytes(), b'{truncated')
            self.assertEqual((receipt.parent / 'previous-unverified-metrics.json').read_bytes(), b'{truncated')
            self.assertTrue(all(r.sha(path) == digest for path, digest in watched.items()))

    def test_wrong_scope_or_original_package_or_payload_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                r.prepare(results, original, study, 'validation-10', '../old', self.runtime_path)
            (original / 'worker.py').write_bytes(b'changed model path')
            with self.assertRaisesRegex(ValueError, 'Original frozen package changed'):
                self.prepare(original, study, results)
            (original / 'worker.py').write_bytes(b'original worker')
            next((profile / 'jobs').glob('*/tracks.npz')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload changed'):
                self.prepare(original, study, results)

    def test_original_evaluator_user_environment_and_byte_identical_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, study, results, evaluator, profile = self.fixture(root)
            previous = profile / 'evaluation/metrics.json'
            previous.write_bytes(b'{partial previous metrics')
            receipt = self.prepare(original, study, results)
            proof = r.read(receipt)
            scratch = root / 'scratch-output'
            scratch.mkdir()
            exact = self.response(receipt)['metrics_json'].encode()
            captured = io.StringIO()
            def official_process(command, cwd, env, check):
                self.assertEqual(command[1], str(evaluator / 'evaluate.py'))
                self.assertEqual(cwd, str(evaluator))
                self.assertEqual(env['HOME'], '/image/original-private-home')
                self.assertEqual(env['PYTHONPATH'], '/image/original-library-path')
                self.assertNotIn('PYTHONUSERBASE', env)
                (scratch / 'metrics.json').write_bytes(exact)
                return SimpleNamespace(returncode=0)
            original_env = {'HOME': '/image/original-private-home', 'PYTHONPATH': '/image/original-library-path'}
            with patch.dict(os.environ, original_env, clear=True), patch.object(r, 'identity', return_value=self.evaluator_user), patch.object(r.importlib, 'import_module', side_effect=lambda n: self.modules[n]), patch.object(r.subprocess, 'run', side_effect=official_process), contextlib.redirect_stdout(captured):
                self.assertEqual(r.evaluate(profile, scratch, evaluator, receipt, results / 'manifest.json'), 0)
            messages = [line[len(r.METRICS_TAG):] for line in captured.getvalue().splitlines() if line.startswith(r.METRICS_TAG)]
            self.assertEqual(len(messages), 1)
            response = root / 'response.json'
            r.save(response, r.json.loads(messages[0]))
            self.assertEqual(previous.read_bytes(), b'{partial previous metrics')
            owner = {'uid': proof['owner_uid'], 'gid': proof['owner_gid']}
            with patch.object(r, 'identity', return_value=owner):
                r.publish(results, receipt, response)
            self.assertEqual(previous.read_bytes(), exact)
            saved = r.read(profile / 'evaluation/permission-repair-receipt.json')
            self.assertTrue(saved['metrics_byte_identical'])
            self.assertEqual(saved['evaluator_identity'], self.evaluator_user)
            self.assertEqual(saved['publisher_identity'], owner)
            self.assertEqual((profile / 'evaluation' / saved['retained_unverified_metrics']).read_bytes(), b'{partial previous metrics')

    def test_owner_user_must_not_compute_and_wrong_dependencies_fail_early(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, study, results, evaluator, profile = self.fixture(root)
            receipt = self.prepare(original, study, results)
            scratch = root / 'scratch'
            scratch.mkdir()
            with patch.object(r, 'identity', return_value={'uid': 997, 'gid': 997}), patch.object(r.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'original image user'):
                    r.evaluate(profile, scratch, evaluator, receipt, results / 'manifest.json')
                child.assert_not_called()
            Path(self.modules['psutil'].__file__).write_text('# changed psutil', encoding='utf-8')
            with patch.object(r, 'identity', return_value=self.evaluator_user), patch.object(r.importlib, 'import_module', side_effect=lambda n: self.modules[n]), patch.object(r.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'dependency changed'):
                    r.evaluate(profile, scratch, evaluator, receipt, results / 'manifest.json')
                child.assert_not_called()

    def test_transport_tamper_and_wrong_case_set_do_not_write_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, study, results, evaluator, profile = self.fixture(root)
            receipt = self.prepare(original, study, results)
            proof = r.read(receipt)
            response_file = root / 'response.json'
            response = self.response(receipt)
            response['metrics_json'] += ' '
            r.save(response_file, response)
            owner = {'uid': proof['owner_uid'], 'gid': proof['owner_gid']}
            with patch.object(r, 'identity', return_value=owner):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    r.publish(results, receipt, response_file)
                bad = self.response(receipt, b'{"results":[],"aggregates":{}}')
                r.save(response_file, bad)
                with self.assertRaisesRegex(ValueError, 'case set mismatch'):
                    r.publish(results, receipt, response_file)
            self.assertFalse((profile / 'evaluation/metrics.json').exists())

    def test_verified_metrics_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            r.save(profile / 'evaluation/verified.json', {'complete': True})
            with self.assertRaisesRegex(ValueError, 'verified evaluator cache'):
                self.prepare(original, study, results)


if __name__ == '__main__':
    unittest.main()
