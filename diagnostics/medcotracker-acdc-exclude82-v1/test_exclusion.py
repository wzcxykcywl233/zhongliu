import copy
import os
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import common as C
import external as E
import reports as R
from exclusion import EXCLUDED, inventory, make_schedule, dataset_class


def fixture_manifest():
    rows = []
    for group, start in zip(('DCM','HCM','MINF','NOR','RV'), range(1, 101, 20)):
        for i in range(20):
            rows.append({'patient': f'patient{start+i:03}', 'group': group,
                         'split': 'external-train-80' if i < 16 else 'external-dev-20'})
    return {'patients': rows}


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        C.save(self.root / 'manifest.json', {'fixture': 1})

    def test_all9000_steps_no082_no_dev_only_original082_replaced(self):
        manifest = fixture_manifest()
        train, dev, _ = inventory(manifest)
        for seed in (20261023, 20261024, 20261025):
            plan = make_schedule(manifest, seed)
            self.assertEqual(len(plan['eligible_train']), 79)
            self.assertEqual(plan['unchanged_dev'], dev)
            self.assertIn(plan['replacement_steps'], (37, 38))
            replaced = []
            for row in plan['rows']:
                self.assertNotEqual(row['patient'], EXCLUDED)
                self.assertNotIn(row['patient'], dev)
                self.assertEqual(row['replaced'], row['original_patient'] == EXCLUDED)
                if row['replaced']:
                    replaced.append(row['patient'])
                    self.assertEqual(train[row['index']]['group'], 'RV')
                else:
                    self.assertEqual(row['original_index'], row['index'])
                with patch.object(E.S, 'SEED', seed):
                    self.assertEqual((row['epoch'], row['original_index'], row['slot']), E.schedule(row['step']))
            self.assertEqual(len(set(replaced[:15])), 15)
            self.assertEqual(len(set(replaced[15:30])), 15)
            self.assertEqual(plan, make_schedule(manifest, seed))
            self.assertEqual(plan['rows'][1375:], make_schedule(manifest, seed)['rows'][1375:])

    def test_loader_never_decodes082_and_preserves_unchanged_sampling(self):
        manifest = fixture_manifest()
        train, _, _ = inventory(manifest)
        decoded = []
        class FakeOriginal:
            def __init__(self, root, path, seed):
                self.manifest, self.rows, self.seed, self.epoch = manifest, train, seed, 0
            def __getitem__(self, index):
                decoded.append(self.rows[index]['patient'])
                return {'patient': self.rows[index]['patient'], 'sampling_seed': self.seed + self.epoch*1000003 + index}
        dataset = dataset_class(FakeOriginal)(None, None, 20261023)
        plan = make_schedule(manifest, 20261023, 240)
        for row in plan['rows']:
            dataset.epoch = row['epoch']
            sample = dataset[row['original_index']]
            self.assertEqual(sample['patient'], row['patient'])
            self.assertEqual(sample['sampling_seed'], 20261023 + row['epoch']*1000003 + row['index'])
        self.assertNotIn(EXCLUDED, decoded)

    def test_invalid_inventory_not_silently_filtered_or_resplit(self):
        original = fixture_manifest()
        for mutate in ('missing', 'duplicate', 'dev', 'group'):
            m = copy.deepcopy(original)
            if mutate == 'missing': m['patients'].pop()
            elif mutate == 'duplicate': m['patients'][-1] = m['patients'][0]
            elif mutate == 'dev': m['patients'][81]['split'] = 'external-dev-20'
            else: m['patients'][81]['group'] = 'NOR'
            with self.assertRaises(ValueError): inventory(m)

    def test_unaffected_label_byte_identical_and_video_mismatch_fails(self):
        video = C.torch.zeros(2, 3, 4, 4)
        sample = {'patient': 'patient001', 'sampling': {'frames': [0,1]}, 'video': video}
        slot = E.schedule(0)[2]
        sig = {'step': 0, 'patient': sample['patient'], 'sampling': sample['sampling'],
               'video_sha256': C.S.tensor_sha(video[None]), 'slot': slot}
        labels = self.root / 'old-labels'
        C.atomic_torch(labels / '0000.pth', {'signature': sig, 'tracks': C.torch.ones(1,2,1,2)})
        before = C.digest(labels / '0000.pth')
        C.save(self.root / 'exclusion-schedule.json', {'rows': [{'patient': sample['patient'], 'slot': slot, 'replaced': False}]})
        with patch.object(E, 'BASELINE_LABELS', labels), patch.object(E, 'HISTORICAL_ROWS', [{'label_sha256': before}]):
            _, copied = E.label_for(self.root, 0, sample, self.root)
            self.assertEqual(before, copied)
            self.assertTrue(C.marker(self.root, 'full/labels/0000.json')['historical_label_byte_identical'])
            sample['video'] = video + 1
            with self.assertRaisesRegex(ValueError, 'clip changed'): E.label_for(self.root, 0, sample, self.root)
        self.assertEqual(before, C.digest(labels / '0000.pth'))

    def test_excluded_sample_rejected_even_if_a_cached_label_exists(self):
        sample = {'patient': EXCLUDED, 'sampling': {}, 'video': C.torch.zeros(2,3,4,4)}
        C.save(self.root / 'exclusion-schedule.json', {'rows': [{'patient': EXCLUDED, 'slot': E.schedule(0)[2]}]})
        with self.assertRaisesRegex(ValueError, 'exclusion schedule'):
            E.label_for(self.root, 0, sample, self.root)

    def test_replacement_smoke_covers_all_teachers_and_freezes_proof(self):
        schedule = make_schedule(fixture_manifest(), C.S.SEED)
        C.save(self.root / 'exclusion-schedule.json', schedule)
        audit = self.root / 'audit'
        C.save(audit / 'data-manifest.json', fixture_manifest())
        plan = {'baseline_binding': {}, 'weights_sha256': {}, 'external_manifest_sha256': C.digest(audit / 'data-manifest.json')}
        observed = []
        class Data:
            def set_epoch(self, epoch): self.epoch = epoch
            def __getitem__(self, index):
                row = next(r for r in schedule['rows'] if r['epoch'] == self.epoch and r['original_index'] == index)
                return {'patient': row['patient']}
        def label(root, step, sample, weights):
            observed.append((step, sample['patient']))
            path = root / 'full/labels' / f'{step:04}.pth'
            values = {k: C.torch.ones(1) for k in ('queries','tracks','visibility','confidence')}
            C.atomic_torch(path, values)
            C.publish(root, f'full/labels/{step:04}.json', [path])
            return values, C.digest(path)
        with patch.object(E, 'gpu'), patch.object(E, 'checked_plan', return_value=plan), \
                patch.object(E, 'baseline_binding', return_value={}), patch.object(E, 'baseline_steps', return_value=[]), \
                patch.object(E, 'dataset_class', return_value=lambda *args, **kwargs: Data()), patch.object(E, 'label_for', side_effect=label):
            E.exclusion_smoke(self.root, self.root, audit, self.root)
            E.exclusion_smoke(self.root, self.root, audit, self.root)
        self.assertEqual(len(observed), 4)
        self.assertNotIn(EXCLUDED, [p for s,p in observed])
        replaced = [schedule['rows'][step] for step,_ in observed if schedule['rows'][step]['replaced']]
        self.assertEqual({r['slot'] for r in replaced}, {0,1,2})
        self.assertEqual(C.marker(self.root, 'smoke/exclusion-verified.json')['actual_replacements'], 3)

    def test_baseline_budget_hashes_and_datasets_bound(self):
        root = self.root / 'baseline'
        fields = ('weights_sha256','images','teacher_pool','external_manifest_sha256','trackrad_ids','trackrad_files','student_config')
        plan = {'rule': copy.deepcopy(C.RULE), **{k: {} for k in fields}}
        old = {**plan, 'rule': {k:v for k,v in C.RULE.items() if k != 'exclusion'}, 'source_sha256': {}}
        C.save(root / 'manifest.json', old)
        C.save(root / 'full/external/B1/steps.json', [{'step': s} for s in range(3000)])
        C.save(root / 'full/external/B1/initial.json', {'state_hashes': {}})
        paths = [root / 'full/external/B1' / n for n in ('steps.json','initial.json')]
        C.publish(root, 'full/external/B1/verified.json', paths, steps=3000, frozen_tensors_exact=True)
        bound = C.baseline_binding(plan, root)
        self.assertEqual(bound['manifest.json'], C.digest(root / 'manifest.json'))
        wrong = copy.deepcopy(plan); wrong['images'] = {'Training': 'other'}
        with self.assertRaisesRegex(ValueError, 'inputs differ'): C.baseline_binding(wrong, root)
        C.save(paths[0], [{'step': 0}])
        with self.assertRaisesRegex(ValueError, 'Artifact changed'): C.baseline_binding(plan, root)

    def test_original_student_report_and_case_pairing(self):
        oldroot = self.root / 'old'
        oldreport = oldroot / 'reports/validation-10'
        oldmanifest = oldroot / 'manifest.json'
        C.save(oldmanifest, {'seed_manifests_sha256': {str(s): str(s) for s in C.SEEDS}})
        values = {s: {'baseline_binding': {'manifest.json': str(s)}, 'trackrad_ids': {'validation-10': ['V_001']}} for s in C.SEEDS}
        current, cases, paths = [], {}, []
        for seed in C.SEEDS:
            for arm in C.ARMS:
                agg = {field: .9 for field in R.FIELDS.values()}
                metric = oldroot / f'seed_{seed}/full/validation-10/{arm}/{C.PROFILE}/metrics.json'
                C.save(metric, {'aggregates': agg, 'results': [{'case_id': 'V_001', **agg}]})
                paths.append(metric)
                delta = .001 if arm == 'B1' else 0.
                current.append({'Seed': seed, 'Arm': arm, **{k: .9 + delta for k in R.FIELDS}})
                cases[seed, arm] = {'V_001': {f: .9 + delta for f in R.FIELDS.values()}}
        C.publish(oldroot, 'reports/validation-10/verified.json', paths, split='validation-10')
        def compare():
            return R.compare_original(self.root, 'validation-10', current, cases, values, oldreport, oldroot, oldmanifest)
        files, paired = compare()
        self.assertTrue(all(p.exists() for p in files))
        self.assertEqual(len(paired), 6)
        self.assertTrue(all(abs(r['Delta_DSC'] - .001) < 1e-10 for r in paired if r['Candidate'] == 'B1_exclude082'))
        self.assertTrue(all(r['Delta_DSC'] == 0 for r in paired if r['Candidate'] == 'R1_repeat'))
        compare()
        C.save(paths[0], {'changed': True})
        with self.assertRaisesRegex(ValueError, 'metrics changed'): compare()


if __name__ == '__main__': unittest.main()
