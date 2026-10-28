import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common as C
import source_provenance as P


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def current_fixture(self, relative, crlf=False):
        data = (C.CHECKOUT / relative).read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(data).hexdigest(), P.REVIEWED[relative][P.AFTER][0])
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.replace(b'\n', b'\r\n') if crlf else data)
        return path

    def test_exact_bytes_require_no_compatibility(self):
        path = self.root / 'model.py'
        path.write_bytes(b'lr = 1e-6\n')
        proof = P.verify_historical_sources(self.root, {'model.py': C.digest(path)}, C.S.safe_file)
        self.assertEqual(proof['checked_files'], 1)
        self.assertEqual(proof['accepted_compatibilities'], [])

    def test_only_pinned_three_file_output_path_transition_allowed(self):
        expected = {}
        for relative, versions in P.REVIEWED.items():
            self.current_fixture(relative, crlf=True)
            expected[relative] = versions[P.BEFORE][0]
        before = {r: C.digest(self.root / r) for r in expected}
        result = P.verify_historical_sources(self.root, expected, C.S.safe_file)
        self.assertEqual(len(result['accepted_compatibilities']), 3)
        self.assertEqual({r['reason'] for r in result['accepted_compatibilities']}, {'reviewed_output_path_fix'})
        self.assertEqual(before, {r: C.digest(self.root / r) for r in expected})

    def test_only_pinned_lf_crlf_variants_allowed(self):
        expected = {}
        for relative, versions in P.REVIEWED.items():
            self.current_fixture(relative)
            expected[relative] = versions[P.AFTER][1]
        result = P.verify_historical_sources(self.root, expected, C.S.safe_file)
        self.assertEqual({r['reason'] for r in result['accepted_compatibilities']}, {'reviewed_LF_CRLF_only'})

    def test_arbitrary_readme_launcher_and_test_edits_still_fail(self):
        expected = {}
        for relative, versions in P.REVIEWED.items():
            path = self.current_fixture(relative)
            path.write_bytes(path.read_bytes() + b'\n# UNREVIEWED EDIT\n')
            expected[relative] = versions[P.BEFORE][0]
        with self.assertRaisesRegex(ValueError, 'unreviewed changes blocked') as caught:
            P.verify_historical_sources(self.root, expected, C.S.safe_file)
        # All differences reported at once, rather than one failure per remote run.
        for relative in expected:
            self.assertIn(relative, str(caught.exception))
        self.assertIn('expected=', str(caught.exception))
        self.assertIn('actual=', str(caught.exception))

    def test_training_code_and_artifacts_stay_byte_exact(self):
        source = self.root / 'external.py'
        source.write_bytes(b'lr = 1e-6\n')
        expected = C.digest(source)
        source.write_bytes(b'lr = 1e-6\r\n')
        with self.assertRaisesRegex(ValueError, 'external.py'):
            P.verify_historical_sources(self.root, {'external.py': expected}, C.S.safe_file)
        source.write_bytes(b'lr = 1e-5\n')
        with self.assertRaisesRegex(ValueError, 'external.py'):
            P.verify_historical_sources(self.root, {'external.py': expected}, C.S.safe_file)
        relative = P.PREFIX + 'README.txt'
        self.current_fixture(relative)
        # The generic verifier is NOT relaxed, even for the allowlisted README.
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            C.verify_files(self.root, {relative: P.REVIEWED[relative][P.BEFORE][0]})
        for name in ('checkpoint.pth', 'labels.pth', 'data-manifest.json'):
            file = self.root / name
            file.write_bytes(b'unchanged\n'); digest = C.digest(file)
            file.write_bytes(b'unchanged\r\n')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                C.verify_files(self.root, {name: digest})

    def test_unknown_expected_hash_missing_file_and_path_escape_fail(self):
        relative = P.PREFIX + 'README.txt'
        self.current_fixture(relative)
        with self.assertRaisesRegex(ValueError, 'unreviewed'):
            P.verify_historical_sources(self.root, {relative: '0'*64}, C.S.safe_file)
        with self.assertRaisesRegex(ValueError, 'missing'):
            P.verify_historical_sources(self.root, {'missing.py': '0'*64}, C.S.safe_file)
        with self.assertRaisesRegex(ValueError, 'escapes root'):
            P.verify_historical_sources(self.root, {'../outside.py': '0'*64}, C.S.safe_file)

    def test_baseline_binding_records_reviewed_mismatches_without_rewriting_old_proof(self):
        baseline = self.root / 'baseline'
        fields = ('weights_sha256','images','teacher_pool','external_manifest_sha256','trackrad_ids','trackrad_files','student_config')
        plan = {'rule': copy.deepcopy(C.RULE), **{key: {} for key in fields}}
        expected = {}
        for relative, versions in P.REVIEWED.items():
            self.current_fixture(relative, crlf=True)
            expected[relative] = versions[P.BEFORE][1]
        old = {**plan, 'rule': {key: val for key, val in C.RULE.items() if key != 'exclusion'},
               'source_sha256': expected}
        C.save(baseline / 'manifest.json', old)
        C.save(baseline / 'full/external/B1/steps.json', [{'step': s} for s in range(3000)])
        C.save(baseline / 'full/external/B1/initial.json', {'state_hashes': {}})
        paths = [baseline / 'full/external/B1' / n for n in ('steps.json','initial.json')]
        C.publish(baseline, 'full/external/B1/verified.json', paths, steps=3000, frozen_tensors_exact=True)
        before = {p.relative_to(baseline).as_posix(): C.digest(p) for p in baseline.rglob('*') if p.is_file()}
        with patch.object(C, 'CHECKOUT', self.root):
            binding = C.baseline_binding(plan, baseline)
        self.assertEqual(len(binding['source_verification']['accepted_compatibilities']), 3)
        self.assertEqual(before, {p.relative_to(baseline).as_posix(): C.digest(p) for p in baseline.rglob('*') if p.is_file()})
        altered = copy.deepcopy(plan); altered['weights_sha256'] = {'model.pth': 'new'}
        with patch.object(C, 'CHECKOUT', self.root), self.assertRaisesRegex(ValueError, 'inputs differ: weights_sha256'):
            C.baseline_binding(altered, baseline)


if __name__ == '__main__': unittest.main()
