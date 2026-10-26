import gzip
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import audit
from qa_core import array_sha, label_metrics, mask_metrics, mask_review_flags, nifti_header, safe, temporal_metrics
from review import raw_previews, trajectory_preview


def header(path, qcode=1, scode=1, sscale=1.):
    b = bytearray(348)
    struct.pack_into('<i', b, 0, 348)
    struct.pack_into('<8h', b, 40, 4, 24, 24, 2, 4, 1, 1, 1)
    struct.pack_into('<8f', b, 76, 1, 1, 1, 1, 1, 0, 0, 0)
    struct.pack_into('<2h', b, 252, qcode, scode)
    struct.pack_into('<12f', b, 280, sscale, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
    b[344:348] = b'n+1\0'
    with gzip.open(path, 'wb') as stream: stream.write(b)


class QualityTests(unittest.TestCase):
    def test_temporally_static_but_textured(self):
        a = np.tile(np.arange(24*24).reshape(1, 1, 24, 24), (5, 2, 1, 1))
        slices, pairs = temporal_metrics(a)
        self.assertIn('temporally_static_slice_review', slices[0]['flags'])
        self.assertNotIn('constant_intensity_slice', slices[0]['flags'])
        self.assertEqual(sum(r['exact_duplicate'] for r in pairs), 8)

    def test_jump_and_native_short_length(self):
        a = np.zeros((12, 1, 24, 24))
        a[6:] = 100
        rows, pairs = temporal_metrics(a)
        self.assertIn(6, rows[0]['jump_frames'])
        self.assertEqual(len(pairs), 11)
        self.assertEqual(rows[0]['frames'], 12)

    def test_loop_endpoint_not_adjacent(self):
        a = np.stack([np.full((1, 24, 24), i) for i in (0, 1, 2, 0)])
        rows, _ = temporal_metrics(a)
        self.assertEqual(rows[0]['unique_frames'], 3)
        self.assertEqual(rows[0]['adjacent_duplicate_frames'], [])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): temporal_metrics(np.full((4, 1, 24, 24), np.nan))

    def test_header_warning_is_not_data_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'a.nii.gz'
            header(path)
            self.assertEqual(nifti_header(path)['flags'], [])
            header(path, sscale=2)
            self.assertIn('qform_sform_disagree_review', nifti_header(path)['flags'])
            self.assertIn('sform_scale_disagrees_with_pixdim_review', nifti_header(path)['flags'])

    def test_absent_slice_mask_allowed(self):
        a = np.zeros((2, 24, 24), dtype='uint8')
        a[0, 4:9, 4:9] = 3
        rows = mask_metrics(a, 'ED', [2, 3, 4])
        r = next(r for r in rows if r['label'] == 3 and r['slice_z'] == 0)
        self.assertEqual(r['area_mm2'], 150)
        a[0, 0, 0] = 4
        with self.assertRaises(ValueError): mask_metrics(a, 'ED', [1, 1, 1])

    def test_internal_mask_gap_review_not_missing_end_slices(self):
        a = np.zeros((5, 24, 24), dtype='uint8')
        a[1, 4:9, 4:9] = 3
        a[3, 4:9, 4:9] = 3
        flags = mask_review_flags(mask_metrics(a, 'ED', [1, 1, 1]))
        self.assertEqual(flags, ['ED_class_3_internal_empty_slice_2_review'])

    def labels(self):
        p = np.full((1, 12, 3, 2), 20., dtype='float32')
        v = np.ones((1, 12, 3), dtype='float32')
        q = np.array([[[0, 20, 20], [0, 20, 20], [0, 20, 20]]], dtype='float32')
        return p, v, v.copy(), q

    def test_static_labels_are_review_only(self):
        r = label_metrics(*self.labels())
        self.assertEqual(r['static_track_fraction'], 1)
        self.assertEqual(r['query_anchor_error_max'], 0)
        self.assertIn('static_track_review_not_bad_by_itself', r['flags'])

    def test_jumps_and_visibility(self):
        p, v, c, q = self.labels()
        p[:, 6:, 0, 0] += 200
        v[:, :, 0] = 0
        r = label_metrics(p, v, c, q)
        self.assertEqual(r['first_jump_frame'], 6)
        self.assertGreater(r['jump_pair_fraction'], 0)
        self.assertEqual(r['visible_jump_count'], 0)

    def test_invalid_labels(self):
        p, v, c, q = self.labels()
        q[0, 0, 0] = 12
        with self.assertRaises(ValueError): label_metrics(p, v, c, q)
        q[0, 0, 0] = 0
        p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): label_metrics(p, v, c, q)

    def test_upstream_half_open_subpixel_query_extent(self):
        p, v, c, q = self.labels()
        q[0, :, 1:] = [511.75, 383.75]
        p[:] = [511.75, 383.75]
        r = label_metrics(p, v, c, q)
        self.assertEqual(r['out_of_bounds_fraction'], 0)
        p[:, 2:, :, 0] = 512
        self.assertGreater(label_metrics(p, v, c, q)['out_of_bounds_fraction'], 0)
        q[0, 0, 1] = 512
        with self.assertRaises(ValueError): label_metrics(p, v, c, q)

    def test_safety_and_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            for rel in ('../x', '/x', 'C:/x', 'a/../../b', 'a\\b'):
                with self.assertRaises(ValueError): safe(td, rel)
            path = Path(td)/'config.json'
            audit.freeze(path, {'a': 1})
            audit.freeze(path, {'a': 1})
            with self.assertRaises(ValueError): audit.freeze(path, {'a': 2})

    def test_previews_all_frames_no_interpolated_masks(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            a = np.random.default_rng(1).normal(size=(12, 3, 24, 24)).astype('float32')
            mask = np.zeros((3, 24, 24), dtype='uint8')
            mask[:, 4:9, 4:9] = 2
            slices, _ = temporal_metrics(a)
            raw_previews(td, a, {'ED': mask, 'ES': mask}, {'ED': 0, 'ES': 7}, slices)
            with Image.open(Path(td)/'cine.gif') as image: self.assertEqual(image.n_frames, 12)
            p, v, _, _ = self.labels()
            trajectory_preview(Path(td)/'track.gif', np.repeat(a[:, :1], 3, axis=1)+100, p, v)
            self.assertTrue((Path(td)/'ED.png').exists())

    @unittest.skipUnless(importlib.util.find_spec('SimpleITK'), 'SimpleITK only available in remote pinned runtime')
    def test_real_nifti_roundtrip_and_patient_audit(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'data/training/patient001'
            source.mkdir(parents=True)
            a = np.random.default_rng(2).integers(0, 100, (12, 2, 24, 24)).astype('int16')
            image = sitk.GetImageFromArray(a, isVector=False)
            sitk.WriteImage(image, str(source/'patient001_4d.nii.gz'))
            for t in (0, 7):
                sitk.WriteImage(sitk.GetImageFromArray(a[t]), str(source/f'patient001_frame{t+1:02}.nii.gz'))
                mask = np.zeros((2, 24, 24), dtype='uint8')
                mask[:, 3:6, 3:6] = 1
                mask[:, 8:10, 8:10] = 2
                mask[:, 11:13, 11:13] = 3
                sitk.WriteImage(sitk.GetImageFromArray(mask), str(source/f'patient001_frame{t+1:02}_gt.nii.gz'))
            row = {'patient': 'patient001', 'split': 'external-train-80', 'group': 'NOR',
                   'cine': 'training/patient001/patient001_4d.nii.gz', 'shape_tzyx': list(a.shape),
                   'annotation_frames_zero_based': {'ED': 0, 'ES': 7},
                   'files_sha256': {p.name: audit.sha(p) for p in source.iterdir()}}
            result = audit.patient_audit(root/'data', row, root/'out/raw/patient001')
            self.assertEqual(result['data_errors'], [])
            self.assertTrue(result['voxel_checks_completed'])
            self.assertIn('identical_ED_ES_masks_review', result['review_flags'])
            self.assertEqual(result, audit.patient_audit(root/'data', row, root/'out/raw/patient001'))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch only needed for cached tensor replay')
    def test_replay_requires_exact_historical_input(self):
        import torch
        import torch.nn.functional as F
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'source'
            source.write_bytes(b'fixture')
            a = np.random.default_rng(4).normal(size=(12, 1, 24, 24)).astype('float32')
            low, high = [float(x) for x in np.percentile(a[:, 0], [1, 99])]
            v = a[:, 0].copy()
            v -= np.float32(low)
            v /= np.float32(high-low)
            np.clip(v, 0, 1, out=v)
            v *= np.float32(255)
            video = F.interpolate(torch.from_numpy(v)[:, None], (384, 512), mode='bilinear', align_corners=False)
            video = video.clamp(0, 255).expand(-1, 3, -1, -1).contiguous()
            p, vis, c, q = self.labels()
            label = dict(zip(('tracks', 'visibility', 'confidence', 'queries'), map(torch.from_numpy, (p, vis, c, q))))
            label['signature'] = {'video_sha256': array_sha(video.numpy()), 'sampling': {
                'slice_z': 0, 'frames': list(range(12)), 'intensity_low': low, 'intensity_high': high}}
            row = {'cine': 'source', 'files_sha256': {'source': audit.sha(source)}}
            with patch.object(audit, 'load_volume', return_value=(a, {})):
                result = audit.replay(root, row, label, root, 0)
                self.assertTrue(result['input_replay_exact'])
                label['signature']['video_sha256'] = 'wrong'
                with self.assertRaises(ValueError): audit.replay(root, row, label, root, 0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_consumed_label_integrity_and_split(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'labels').mkdir()
            p = torch.full((1, 12, 128, 2), 20.)
            v = torch.ones((1, 12, 128))
            q = torch.cat((torch.zeros(1, 128, 1), torch.full((1, 128, 2), 20.)), -1)
            sampling = {'source_shape_tzyx': [12, 2, 24, 24], 'slice_z': 0, 'frames': list(range(12))}
            sig = {'step': 0, 'slot': 2, 'patient': 'patient001', 'sampling': sampling}
            label = {'signature': sig, 'tracks': p, 'visibility': v, 'confidence': v, 'queries': q}
            path = root/'labels/0000.pth'
            torch.save(label, path)
            h = audit.sha(path)
            proof = {'complete': True, 'manifest_sha256': 'frozen', 'signature': sig,
                     'files_sha256': {'full/labels/0000.pth': h}}
            audit.save(path.with_suffix('.json'), proof)
            step = {'step': 0, 'patient': 'patient001', 'slot': 2, 'label_sha256': h}
            patients = {'patient001': {'split': 'external-train-80', 'shape_tzyx': [12, 2, 24, 24]}}
            result = audit.validate_cached_label(root, step, 'frozen', patients)
            self.assertEqual(result['queries'], 128)
            patients['patient001']['split'] = 'external-dev-20'
            with self.assertRaises(ValueError): audit.validate_cached_label(root, step, 'frozen', patients)
            patients['patient001']['split'] = 'external-train-80'
            step['label_sha256'] = 'bad'
            with self.assertRaises(ValueError): audit.validate_cached_label(root, step, 'frozen', patients)
            step['label_sha256'] = h
            proof['signature']['slot'] = 1
            audit.save(path.with_suffix('.json'), proof)
            with self.assertRaises(ValueError): audit.validate_cached_label(root, step, 'frozen', patients)

    def test_manifest_requires_frozen_80_20(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [{'patient': f'patient{i:03}', 'split': 'external-dev-20' if i <= 20 else 'external-train-80',
                     'cine': f'training/patient{i:03}/patient{i:03}_4d.nii.gz', 'files_sha256': {}} for i in range(1, 101)]
            value = {'audit_version': 2, 'training_ready': True, 'patients': rows}
            audit.save(root/'data-manifest.json', value)
            self.assertEqual(len(audit.manifest(root)['patients']), 100)
            rows[0]['split'] = 'external-train-80'
            audit.save(root/'data-manifest.json', value)
            with self.assertRaises(ValueError): audit.manifest(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required')
    def test_complete_label_report_counts_and_no_quality_certification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mpath = root/'external-manifest.json'
            audit.save(mpath, {'fixture': 'frozen 80/20'})
            m = {'_path': str(mpath), 'patients': [{'patient': 'patient001'}]}
            for seed in range(3):
                source = root/'cache'/f'seed_{seed}'
                source.mkdir(parents=True)
                plan = {'external_manifest_sha256': audit.sha(mpath), 'rule': {'external_steps': 3000, 'seed': seed},
                        'teacher_pool': [{'family': name} for name in ('v2', 'on3', 'offscaled')]}
                audit.save(source/'manifest.json', plan)
                audit.save(source/'steps.json', [{'step': step, 'slot': step % 3} for step in range(3000)])
                audit.save(source/'verified.json', {'complete': True, 'manifest_sha256': audit.sha(source/'manifest.json'),
                           'files_sha256': {'full/external/B1/steps.json': audit.sha(source/'steps.json')}})
            def fake_validator(source, st, mh, patients, pool):
                return {'patient': 'patient001', 'slot': st['slot'], 'slice_z': 0,
                        'visible_fraction': 1., 'confidence_mean': 1., 'out_of_bounds_fraction': 0.,
                        'static_track_fraction': .5, 'jump_pair_fraction': 0., 'query_anchor_error_max': 0.}
            with patch.object(audit, 'validate_cached_label', side_effect=fake_validator), patch('builtins.print'):
                summary = audit.label_stage(root/'data', root/'cache', m, root/'out', 0)
                audit.report(root/'out')
            self.assertEqual(summary['labels_attempted'], 9000)
            self.assertEqual(summary['invalid_labels_or_proofs'], 0)
            self.assertFalse(audit.read(root/'out/summary.json')['quality_certified'])
            self.assertTrue((root/'out/index.html').exists())

    def test_receipt_detects_modified_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root/'data/training/patient001'
            data.mkdir(parents=True)
            (data/'cine').write_bytes(b'broken')
            row = {'patient': 'patient001', 'split': 'external-train-80', 'group': 'NOR',
                   'cine': 'training/patient001/cine', 'files_sha256': {'cine': audit.sha(data/'cine')}}
            folder = root/'out/raw/patient001'
            with patch.object(audit, 'load_volume', side_effect=ValueError('bad image')):
                result = audit.patient_audit(root/'data', row, folder)
            self.assertFalse(result['voxel_checks_completed'])
            self.assertTrue(result['data_errors'])
            (folder/'result.json').write_text('{}')
            with self.assertRaises(ValueError): audit.patient_audit(root/'data', row, folder)


if __name__ == '__main__': unittest.main()
