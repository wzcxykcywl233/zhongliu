import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import acdc_data as d
from cine_dataset import sample_cine


class PreparationTests(unittest.TestCase):
    def test_stratified_patient_split(self):
        rows = [{'patient': f'patient{i+1:03}', 'group': d.GROUPS[i//20]} for i in range(100)]
        result = d.split_patients(rows)
        self.assertEqual(result, d.split_patients(rows[::-1]))
        self.assertEqual(sum(v == 'external-train-80' for v in result.values()), 80)
        for group in d.GROUPS:
            self.assertEqual(sum(result[r['patient']] == 'external-dev-20' for r in rows if r['group'] == group), 4)

    def test_complete_manifest_retains_short_patient_and_stratification(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root, output = Path(tmp)/'data', Path(tmp)/'output'
            rows = []
            for i in range(100):
                patient = f'patient{i+1:03}'
                (root/'training'/patient).mkdir(parents=True)
                frames = 15 if patient == 'patient011' else 30
                rows.append({'patient': patient, 'group': d.GROUPS[i//20], 'shape_tzyx': [frames, 3, 32, 32],
                             'clip_frames': min(16, frames), 'short_native_cine': frames < 16})
            d.save(root/'download-index.json', {'files': []})
            d.save(root/'download-complete.json', {'complete': True, 'download_index_sha256': d.digest(root/'download-index.json')})
            with patch.object(d, 'audit_patient', side_effect=rows), patch('builtins.print'):
                d.prepare(root, output)
            manifest = d.read(output/'data-manifest.json')
            self.assertEqual(manifest['audit_version'], 2)
            self.assertEqual(manifest['split_counts'], {'external-train-80': 80, 'external-dev-20': 20})
            self.assertEqual(manifest['native_frame_counts'], {'15': 1, '30': 99})
            self.assertEqual(len(manifest['patients']), 100)
            self.assertEqual(manifest['short_native_cines'][0]['patient'], 'patient011')
            self.assertEqual(manifest['clip'], d.clip_policy())

    def test_incomplete_and_duplicate_rejected(self):
        rows = [{'patient': f'patient{i+1:03}', 'group': d.GROUPS[i//20]} for i in range(100)]
        for wrong in (rows[:99], rows[:99] + [rows[0]]):
            with self.assertRaises(ValueError): d.split_patients(wrong)

    def test_group_not_inferred_from_patient_number(self):
        info = d.parse_info('ED: 1\nES: 12\nGroup: RV\nNbFrame: 30\n')
        self.assertEqual(info['Group'], 'RV')
        with self.assertRaises(ValueError): d.parse_info('ED: 1\nGroup: unknown')
        with self.assertRaises(ValueError): d.parse_info('Group: RV\nGroup: NOR')

    def test_no_temporal_loop_or_spatial_video(self):
        # Every time increment is visible in the returned intensity at fixed z.
        array = np.arange(30, dtype=np.float32)[:, None, None, None] + np.zeros((30, 4, 32, 32), np.float32)
        a, proof = sample_cine(array, 1, resize=(32, 32))
        b, proof2 = sample_cine(array, 1, resize=(32, 32))
        self.assertEqual(a.shape, (16, 3, 32, 32))
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(proof, proof2)
        self.assertEqual(np.diff(proof['frames']).tolist(), [1]*15)
        self.assertTrue((a[1:, 0, 0, 0] > a[:-1, 0, 0, 0]).all())
        self.assertTrue(torch.equal(a[:, 0], a[:, 2]))

    def test_bad_clip_rejected(self):
        for array in (np.zeros((1, 2, 32, 32)), np.zeros((30, 2, 32, 32)), np.full((30, 2, 32, 32), np.nan)):
            with self.assertRaises(ValueError): sample_cine(array, 1)

    def test_short_native_cine_uses_every_frame_once(self):
        for frames in (2, 12, 15, 16):
            array = np.arange(frames, dtype=np.float32)[:, None, None, None] + np.zeros((frames, 3, 32, 32), np.float32)
            a, proof = sample_cine(array, 7, resize=(32, 32))
            b, repeated = sample_cine(array, 7, resize=(32, 32))
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(proof, repeated)
            self.assertEqual(list(a.shape), [frames, 3, 32, 32])
            self.assertEqual(proof['frames'], list(range(frames)))
            self.assertEqual(proof['requested_frames'], 16)
            self.assertEqual(proof['actual_frames'], frames)
            self.assertEqual(proof['short_native_cine'], frames < 16)

    def test_invalid_requested_frame_count(self):
        for requested in (0, 1, True, 16.0):
            with self.assertRaises(ValueError): d.clip_length(30, requested)

    def test_long_cine_sampling_and_pixels_unchanged(self):
        import torch.nn.functional as F
        array = np.random.default_rng(17).uniform(0, 500, (30, 3, 32, 32)).astype('float32')
        rng = np.random.default_rng(7)
        z, start = int(rng.integers(3)), int(rng.integers(15))
        plane = array[:, z].astype('float32')
        low, high = (float(x) for x in np.percentile(plane, [1, 99]))
        old = plane[start:start+16].copy()
        old -= np.float32(low)
        old /= np.float32(high-low)
        np.clip(old, 0, 1, out=old)
        old *= np.float32(255)
        old = F.interpolate(torch.from_numpy(old)[:, None], size=(48, 64), mode='bilinear', align_corners=False)
        old = old.clamp(0, 255).expand(-1, 3, -1, -1).contiguous()
        actual, proof = sample_cine(array, 7, resize=(48, 64))
        self.assertTrue(torch.equal(old, actual))
        self.assertEqual(proof['frames'], list(range(start, start+16)))

    def test_paths(self):
        root = Path(tempfile.gettempdir()) / 'acdc-test'
        for path in ('../oops', '/oops', 'x\\y', 'C:/x', 'x//y'):
            with self.assertRaises(ValueError): d.safe_path(root, path)
        self.assertEqual(d.safe_path(root, 'training/patient001/Info.cfg'), (root/'training/patient001/Info.cfg').resolve())

    def test_terms_gate_before_network(self):
        with patch.object(d, 'make_index') as index:
            with self.assertRaises(ValueError): d.download(Path('unused'), 100, False)
            index.assert_not_called()

    def test_file_receipt(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root = Path(tmp)
            path = root/'training/patient001/Info.cfg'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'abc')
            row = {'path': 'training/patient001/Info.cfg', 'file_id': 'abc', 'bytes': 3}
            with self.assertRaises(ValueError): d.download_file(root, row)
            d.save(root/'receipts/abc.json', {'source': row, 'sha256': d.digest(path)})
            with patch.object(d, 'request') as req:
                d.download_file(root, row)
                req.assert_not_called()
            path.write_bytes(b'xyz')
            with self.assertRaises(ValueError): d.download_file(root, row)

    def test_formal_train_gate(self):
        from cine_dataset import ACDCCineDataset
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            path = Path(tmp)/'manifest.json'
            d.save(path, {'schema': d.SCHEMA, 'training_ready': False})
            with self.assertRaises(ValueError): ACDCCineDataset(tmp, path)

    def test_legacy_manifest_requires_new_audit(self):
        from cine_dataset import ACDCCineDataset
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            path = Path(tmp)/'manifest.json'
            d.save(path, {'schema': d.SCHEMA, 'training_ready': True, 'clip': {'frames': 16}})
            with self.assertRaisesRegex(ValueError, 'Re-audit'): ACDCCineDataset(tmp, path)

    def test_audit_reports_exact_failure_not_generic_short_cine(self):
        with self.assertRaisesRegex(ValueError, 'got 1'): d.clip_length(1)
        self.assertEqual(d.clip_length(15), 15)
        self.assertEqual(d.clip_length(30), 16)

    def test_short_download_retries_and_commits_only_complete_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root = Path(tmp)
            row = {'path': 'training/patient001/test.nii.gz', 'file_id': 'abc', 'bytes': 3}
            with patch.object(d, 'request', side_effect=[io.BytesIO(b'a'), io.BytesIO(b'abc')]) as req, patch.object(d.time, 'sleep'):
                d.download_file(root, row)
                self.assertEqual(req.call_count, 2)
            self.assertEqual((root/row['path']).read_bytes(), b'abc')
            self.assertTrue((root/'receipts/abc.json').exists())

    def test_failed_download_cannot_be_marked_complete(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            root = Path(tmp)
            row = {'path': 'bad.nii.gz', 'file_id': 'abc', 'bytes': 3}
            with patch.object(d, 'request', side_effect=[io.BytesIO(b'a') for _ in range(4)]), patch.object(d.time, 'sleep'):
                with self.assertRaises(OSError): d.download_file(root, row)
            self.assertFalse((root/'bad.nii.gz').exists())
            self.assertFalse((root/'receipts/abc.json').exists())
            self.assertTrue((root/'bad.nii.gz.partial').exists())

    def test_four_dimensional_geometry_and_phase_index(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            folder = Path(tmp)/'patient001'
            folder.mkdir()
            (folder/'Info.cfg').write_text('ED: 1\nES: 12\nGroup: DCM\nNbFrame: 28\n')
            array = np.random.default_rng(1).normal(size=(28, 3, 32, 32)).astype('float32')
            cine = sitk.GetImageFromArray(array, isVector=False)
            cine.SetSpacing((1.4, 1.4, 8., .04))
            sitk.WriteImage(cine, str(folder/'patient001_4d.nii.gz'))
            for t in (1, 12):
                img = sitk.GetImageFromArray(array[t-1])
                img.SetSpacing((1.4, 1.4, 8.))
                mask = sitk.GetImageFromArray((array[t-1] > 0).astype('uint8'))
                mask.CopyInformation(img)
                sitk.WriteImage(img, str(folder/f'patient001_frame{t:02}.nii.gz'))
                sitk.WriteImage(mask, str(folder/f'patient001_frame{t:02}_gt.nii.gz'))
            audit = d.audit_patient(folder)
            self.assertEqual(audit['shape_tzyx'], [28, 3, 32, 32])
            self.assertEqual(audit['annotation_frames_zero_based'], {'ED': 0, 'ES': 11})
            (folder/'Info.cfg').write_text('ED: 1\nES: 13\nGroup: DCM\nNbFrame: 28\n')
            with self.assertRaises(Exception): d.audit_patient(folder)

    def test_patient011_style_fifteen_frame_cine_passes(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP')) as tmp:
            folder = Path(tmp)/'patient011'
            folder.mkdir()
            (folder/'Info.cfg').write_text('ED: 1\nES: 8\nGroup: DCM\nNbFrame: 15\n')
            array = np.random.default_rng(11).normal(size=(15, 4, 32, 32)).astype('float32')
            sitk.WriteImage(sitk.GetImageFromArray(array, isVector=False), str(folder/'patient011_4d.nii.gz'))
            for t in (1, 8):
                sitk.WriteImage(sitk.GetImageFromArray(array[t-1]), str(folder/f'patient011_frame{t:02}.nii.gz'))
                sitk.WriteImage(sitk.GetImageFromArray((array[t-1] > 0).astype('uint8')),
                                str(folder/f'patient011_frame{t:02}_gt.nii.gz'))
            audit = d.audit_patient(folder)
            self.assertEqual(audit['shape_tzyx'][0], 15)
            self.assertEqual(audit['clip_frames'], 15)
            self.assertTrue(audit['short_native_cine'])
            self.assertTrue(audit['time_axis_verified_against_ed_es'])
            (folder/'Info.cfg').write_text('ED: 1\nES: 8\nGroup: DCM\nNbFrame: 30\n')
            with self.assertRaisesRegex(ValueError, 'frame count'): d.audit_patient(folder)


if __name__ == '__main__':
    unittest.main()
