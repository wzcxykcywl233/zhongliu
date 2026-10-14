import hashlib
import os
from pathlib import Path
import subprocess
import sysconfig
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
        return original, study, results, evaluator, profile

    def test_manifest_jobs_original_package_preserved_and_previous_metrics_snapshotted(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            watched = {p: r.sha(p) for base in (original, profile / 'jobs') for p in base.rglob('*') if p.is_file()}
            watched[results / 'manifest.json'] = r.sha(results / 'manifest.json')
            metrics = profile / 'evaluation/metrics.json'
            metrics.write_bytes(b'{truncated')
            receipt = r.prepare(results, original, study, 'validation-10', 'old')
            proof = r.read(receipt)
            self.assertEqual(len(proof['cases']), 10)
            self.assertEqual(proof['owner_uid'], (profile / 'evaluation').stat().st_uid)
            self.assertEqual(metrics.read_bytes(), b'{truncated')
            self.assertEqual((receipt.parent / 'previous-unverified-metrics.json').read_bytes(), b'{truncated')
            self.assertTrue(all(r.sha(path) == digest for path, digest in watched.items()))
            pointer = r.read(receipt.parent.parent / 'latest.json')
            self.assertEqual(pointer['sha256'], r.sha(receipt))

    def test_wrong_scope_or_original_package_or_payload_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                r.prepare(results, original, study, 'validation-10', '../old')
            (original / 'worker.py').write_bytes(b'changed model path')
            with self.assertRaisesRegex(ValueError, 'Original frozen package changed'):
                r.prepare(results, original, study, 'validation-10', 'old')
            (original / 'worker.py').write_bytes(b'original worker')
            target = next((profile / 'jobs').glob('*/tracks.npz'))
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload changed'):
                r.prepare(results, original, study, 'validation-10', 'old')

    def test_repaired_entrypoint_keeps_official_code_and_user_dependency_path(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            previous = profile / 'evaluation/metrics.json'
            previous.write_bytes(b'{partial previous metrics')
            receipt = r.prepare(results, original, study, 'validation-10', 'old')
            proof = r.read(receipt)
            output = profile / 'evaluation'
            def official_process(command, cwd, env, check):
                self.assertEqual(command[1], str(evaluator / 'evaluate.py'))
                self.assertEqual(cwd, str(evaluator))
                self.assertEqual(env['HOME'], '/home/user')
                self.assertEqual(env['PYTHONUSERBASE'], '/home/user/.local')
                self.assertIn(sysconfig.get_path('purelib', scheme='posix_user', vars={'userbase': '/home/user/.local'}), env['PYTHONPATH'])
                r.save(output / 'metrics.json', {'official_fixture': 1})
                return SimpleNamespace(returncode=0)
            with patch.object(r, 'identity', return_value={'uid': proof['owner_uid'], 'gid': proof['owner_gid']}), patch.object(r.subprocess, 'run', side_effect=official_process):
                result = r.evaluate(profile, output, evaluator, receipt, results / 'manifest.json')
            self.assertEqual(result, 0)
            self.assertEqual(list(output.glob('.permission-probe-*')), [])
            self.assertFalse(r.read(output / 'permission-repair-receipt.json')['numerical_changes'])
            self.assertEqual(r.read(output / 'permission-repair-receipt.json')['official_metrics_sha256'], r.sha(output / 'metrics.json'))
            retained = r.read(output / 'permission-repair-receipt.json')['retained_unverified_metrics']
            self.assertEqual((output / retained).read_bytes(), b'{partial previous metrics')

    def test_write_failure_or_source_change_stops_before_metrics_computation(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            receipt = r.prepare(results, original, study, 'validation-10', 'old')
            proof = r.read(receipt)
            output = profile / 'evaluation'
            with patch.object(r, 'identity', return_value={'uid': proof['owner_uid'], 'gid': proof['owner_gid']}), patch.object(r, 'write_probe', side_effect=PermissionError('fixture denial')), patch.object(r.subprocess, 'run') as child:
                with self.assertRaises(PermissionError):
                    r.evaluate(profile, output, evaluator, receipt, results / 'manifest.json')
                child.assert_not_called()
            (evaluator / 'evaluate.py').write_bytes(b'wrong scoring code')
            with patch.object(r, 'identity', return_value={'uid': proof['owner_uid'], 'gid': proof['owner_gid']}), patch.object(r.subprocess, 'run') as child:
                with self.assertRaisesRegex(ValueError, 'evaluator source changed'):
                    r.evaluate(profile, output, evaluator, receipt, results / 'manifest.json')
                child.assert_not_called()

    def test_verified_metrics_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            original, study, results, evaluator, profile = self.fixture(Path(temp))
            r.save(profile / 'evaluation/verified.json', {'complete': True})
            with self.assertRaisesRegex(ValueError, 'verified evaluator cache'):
                r.prepare(results, original, study, 'validation-10', 'old')


if __name__ == '__main__':
    unittest.main()
