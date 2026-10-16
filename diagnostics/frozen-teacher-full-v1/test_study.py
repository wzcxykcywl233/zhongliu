"""CPU tests; fixtures are not patient data. No training/Docker/network."""
import copy
import csv
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
APP = PACKAGE.parents[1] / 'cotracker-algorithm'
os.environ['COTRACKER_APP_DIR'] = str(APP)
SPEC = importlib.util.spec_from_file_location('frozen_teacher_full_study', PACKAGE / 'study.py')
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def metric_arm(folder, profile, cases, dsc, cfg=None):
    aggregates = {v: value for v, value in zip(study.FIELDS.values(), (dsc, 4.2, 1.6, 1.8, .84))}
    aggregates['total_time'] = 100.
    rows = [{'case_id': case, **{k: v for k, v in aggregates.items() if k != 'total_time'}} for case in cases]
    write(folder / profile / 'metrics.json', {'results':rows, 'aggregates':aggregates})
    for case in cases:
        job = folder / profile / 'checkpoint/jobs' / case
        output = job / 'output/images/mri-linac-series-targets/output.mha'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'fixture-output-not-mha')
        write(job / 'diagnostics.json', {'profile':profile, 'config':cfg or {},
               'model':{'backbone_architecture':'base','parameters':25385700},
               'prediction':{'output_file_sha256':study.file_sha(output)}})
        (job / '.complete').write_text('complete')
    return study.file_sha(folder / profile / 'metrics.json')


class FrozenTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.prior, self.root = self.base / 'prior', self.base / 'result'
        self.checkout, self.weights, self.data = self.base / 'checkout', self.base / 'weights', self.base / 'datasets'
        self.package = self.base / 'package'
        for path in (self.prior,self.root,self.checkout,self.weights,self.package):
            path.mkdir()
        (self.package / 'study.py').write_text('# package fixture')
        source = self.checkout / 'scripts/sentinel.ps1'
        source.parent.mkdir(); source.write_text('# original source')
        weight_rows = []
        for name in ('scaled_offline.pth','baseline_offline.pth','baseline_online.pth','cotracker2v1.pth'):
            path = self.weights / name
            path.write_bytes(name.encode())
            weight_rows.append({'Name':name,'SHA256':study.file_sha(path)})
        inventory, ids = [], {}
        for split, alias, prefix, count in (('train-40','train','A',40),('validation-10','validation','B',10),('test-38','test','C',38)):
            ids[split] = [f'{prefix}_{i:03}' for i in range(1,count+1)]
            for case in ids[split]:
                path = self.data / alias / case / 'images' / (case + '_frames.mha')
                path.parent.mkdir(parents=True)
                path.write_bytes(b'fixture-not-medical-data')
                inventory.append({'Split':split,'Case':case,'File':f'C:\\frozen\\{alias}\\{case}\\images\\{path.name}', 'SHA256':study.file_sha(path)})
        self.ids = ids
        protocol = {'FullSteps':1000,'FullSeeds':[0,1,2], 'InferenceProfile':study.OLD_PROFILE,
                    'SequenceLength':10,'QueryPoints':384,'TrainIterations':4,'LearningRate':.00005,
                    'BaseSeed':study.BASE_SEED,'ConfidenceTarget':'hard','AuxiliaryWeight':0,'MaskWeight':0,
                    'TeacherPool':['cotracker2v1','online_cotracker_three','offline_cotracker_three'],
                    'Source':[{'File':'scripts/sentinel.ps1','SHA256':study.file_sha(source)}],
                    'Weights':weight_rows,'Dataset':inventory,'MaxMedianStepSeconds':15,'ArmTimeoutSeconds':7200}
        write(self.prior / 'frozen-study.json', protocol)
        images = {k:'sha256:' + digit*64 for k,digit in (('Training','a'),('Inference','b'),('Evaluation','c'))}
        write(self.prior / 'local-runtime-images.json', images)
        self.references = {'trackrad-algorithm-cotracker-algorithm':images['Inference'],'trackrad-evaluation':images['Evaluation']}
        write(self.prior / 'reference-images.json', self.references)
        for name in ('gpu-teacher-proof.json','gpu-student-inference-proof.json','training-audit.json'):
            write(self.prior / 'smoke' / name, {'complete':True})
        write(self.prior / 'pilot/training-audit.json', {'complete':True})
        metric_hashes = {}
        dscs = {'pretrained':.9039725494790238,'old':.8894246089799295,'scaled':.8929203470009137,
                'grid0':.8927349767201612,'iter6':.8935057911480992}
        for policy, score in dscs.items():
            relative = 'pretrained' if policy == 'pretrained' else 'seed_0/' + policy
            folder = self.prior / 'pilot/validation-10' / relative
            digest = metric_arm(folder, study.OLD_PROFILE, ids['validation-10'], score)
            write(folder / 'frozen-images.json', self.references)
            metric_hashes[f'{relative}/{study.OLD_PROFILE}/metrics.json'] = digest
        write(self.prior / 'pilot/validation-10/teacher-student-audit.json', {
            'complete':True,'split':'validation-10','cases':10,'seeds':[0],'steps':125,
            'inference_profile':study.OLD_PROFILE,'policies':['old','scaled','grid0','iter6'],'metrics_sha256':metric_hashes})

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        # The old study's checksum verifier is covered by its own tests. Here
        # isolate selection/source/dataset verification, not fake checkpoints.
        with patch.object(study, 'verify_training') as verify:
            result = study.prepare(self.prior,self.root,self.checkout,self.package,self.weights,self.data)
            verify.assert_called_once_with(self.prior / 'pilot',[0],125)
            return result

    def test_freezes_iter6_despite_degradation_and_preserves_old_evidence(self):
        before = {p.relative_to(self.prior).as_posix():study.file_sha(p) for p in self.prior.rglob('*') if p.is_file()}
        selection = self.prepare()
        self.assertEqual(selection['selected_policy'], 'iter6')
        self.assertTrue(selection['selected_pilot_below_pretrained'])
        self.assertTrue(selection['user_requested_full_run_despite_pilot_degradation'])
        self.assertEqual(selection['teacher_iterations'], 6)
        self.assertEqual(selection['teacher_support_points'], 0)
        self.assertEqual(selection['student_inference_config']['border_points'], 250)
        self.assertEqual(selection['training']['query_points'], 384)
        self.assertEqual(selection['training']['steps'], 1000)
        self.assertIn('NOT a pilot-trained', selection['student_initialization'])
        self.assertEqual(len(list((self.root / 'train').rglob('run-config.json'))), 6)
        self.assertFalse(list(self.root.rglob('*.pth')))
        self.assertEqual(before, {p.relative_to(self.prior).as_posix():study.file_sha(p) for p in self.prior.rglob('*') if p.is_file()})
        self.prepare()  # Exact resume is allowed.

    def test_selection_is_validation_dsc_not_teacher_test_rank(self):
        scores = {p:{k:1. for k in study.FIELDS} for p in ('pretrained','old','scaled','grid0','iter6')}
        scores['pretrained']['DSC'] = 9.
        scores['iter6']['DSC'] = 2.
        scores['scaled']['D98'] = 99.
        self.assertEqual(study.choose_teacher(scores), 'iter6')
        scores['grid0']['DSC'] = 2.
        scores['grid0']['HD95'] = .5
        self.assertEqual(study.choose_teacher(scores), 'grid0')

    def test_original_source_change_is_not_silently_accepted(self):
        (self.checkout / 'scripts/sentinel.ps1').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.prepare()

    def test_weight_or_dataset_inventory_change_stops(self):
        (self.data / 'test/C_001/unrecorded.txt').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            self.prepare()

    def test_package_change_requires_new_output(self):
        self.prepare()
        (self.package / 'study.py').write_text('# different package')
        with self.assertRaisesRegex(ValueError, 'Frozen selection/protocol changed'):
            self.prepare()

    def test_profile_is_same_best_250_configuration(self):
        cfg = study.student_config()
        self.assertEqual(cfg['border_points'], 250)
        self.assertEqual(cfg['n_iterations'], 2)  # Teacher six != student inference six.
        self.assertFalse(cfg['mamba_refiner'])
        self.assertEqual(cfg['query_memory_mode'], 'topk_confidence_diversity')

    def test_output_reader_rejects_wrong_point_count_and_changed_outputs(self):
        folder = self.root / 'validation-10/pretrained'
        cfg = study.student_config()
        metric_arm(folder, study.PROFILE, self.ids['validation-10'], .90, cfg)
        aggregate, rows, digest, outputs = study.load_student_arm(folder,self.ids['validation-10'],cfg)
        self.assertEqual(aggregate['DSC'], .9)
        self.assertEqual(len(outputs), 10)
        job = folder / study.PROFILE / 'checkpoint/jobs/B_001'
        d = study.read(job / 'diagnostics.json')
        d['config']['border_points'] = 1000
        write(job / 'diagnostics.json',d)
        with self.assertRaisesRegex(ValueError, 'frozen best configuration'):
            study.load_student_arm(folder,self.ids['validation-10'],cfg)
        d['config'] = cfg; write(job / 'diagnostics.json',d)
        (job / 'output/images/mri-linac-series-targets/output.mha').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            study.load_student_arm(folder,self.ids['validation-10'],cfg)

    def test_artifact_paths_cannot_escape_input(self):
        for path in ('../other', '/absolute', 'C:\\elsewhere', 'folder/../../other'):
            with self.assertRaises(ValueError):
                study.within(self.root,path)

    def test_report_and_test_gate_keep_degraded_results_and_all_seeds(self):
        self.prepare()
        expected = self.ids['validation-10']
        cfg = study.student_config()
        metric_arm(self.root / 'validation-10/pretrained', study.PROFILE, expected, .904, cfg)
        write(self.root / 'validation-10/pretrained/frozen-images.json', self.references)
        for seed in study.SEEDS:
            for policy in study.POLICIES:
                metric_arm(self.root / f'validation-10/seed_{seed}/{policy}', study.PROFILE, expected,
                           (.89 if policy == 'old' else .894)+seed*.0001, cfg)
                write(self.root / f'validation-10/seed_{seed}/{policy}/frozen-images.json', self.references)
        with patch.object(study, 'checked_training'):
            study.report(self.root,'validation-10')
            study.test_gate(self.root)
        with (self.root / 'validation-10/frozen-teacher-student-summary.csv').open(encoding='utf-8-sig',newline='') as stream:
            summary = {r['Policy']:r for r in csv.DictReader(stream)}
        self.assertEqual(summary['pretrained']['N'],'1')
        self.assertEqual(summary['iter6']['N'],'3')
        self.assertLess(float(summary['iter6']['DSC_Mean']),float(summary['pretrained']['DSC_Mean']))
        shortlist = study.read(self.root / 'test-shortlist.json')
        self.assertEqual(shortlist['policies'],['old','iter6'])
        proof = study.read(self.root / 'validation-10/audit.json')
        self.assertEqual(len(proof['metrics_sha256']),7)
        with patch.object(study, 'checked_training'):
            next((self.root / 'validation-10').rglob('output.mha')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                study.test_gate(self.root)

    def test_audit_rejects_nonoffline_label_changes(self):
        self.prepare()
        def fake_rows(root, seed, policy):
            rows = {i:{'step':i,'case_names':['A_001'],'queries_sha256':'a'*64,'video_sha256':'b'*64,
                       'primary_index':i%3,'auxiliary_index':None,'step_seconds':.4,
                       'label_audit':{'support_draw_sha256':'c'*64,'query_time_counts':[384]+[0]*9,
                                      'tracks_sha256':'d'*64,'visibility_sha256':'e'*64,'confidence_sha256':'f'*64,
                                      'teacher_seconds':.1}} for i in range(1000)}
            if policy == 'iter6':
                rows[0]['label_audit']['tracks_sha256'] = 'f'*64
            folder = root / f'train/seed_{seed}/{policy}'
            (folder / 'cotracker_three_final.pth').write_bytes(b'fixture')
            (folder / 'paired-steps.jsonl').write_text('fixture')
            write(folder / 'initialization-report.json', {'fixture':True})
            return rows
        with patch.object(study,'verify_arm',side_effect=fake_rows):
            with self.assertRaisesRegex(ValueError,'unmodified teacher labels'):
                study.audit(self.root)

    def test_full_paired_audit_passes_and_freezes_all_six_checkpoints(self):
        self.prepare()
        def fake_rows(root, seed, policy):
            sampler = study.TeacherPairSampler(3,seed=study.BASE_SEED+seed)
            rows = {}
            for i in range(1000):
                slot = sampler.sample().primary
                rows[i] = {'step':i, 'case_names':['A_001'], 'queries_sha256':'a'*64, 'video_sha256':'b'*64,
                           'primary_index':slot, 'auxiliary_index':None, 'step_seconds':.4,
                           'label_audit':{'support_draw_sha256':'c'*64,'query_time_counts':[384]+[0]*9,
                                          'tracks_sha256':('a' if slot==2 and policy=='iter6' else 'd')*64,
                                          'visibility_sha256':'e'*64,'confidence_sha256':'f'*64,'teacher_seconds':.1}}
            folder = root / f'train/seed_{seed}/{policy}'
            (folder / 'cotracker_three_final.pth').write_bytes(f'fixture {seed} {policy}'.encode())
            (folder / 'paired-steps.jsonl').write_text('fixture')
            write(folder / 'initialization-report.json', {'fixture':True})
            return rows
        with patch.object(study,'verify_arm',side_effect=fake_rows):
            study.audit(self.root)
        result = study.checked_training(self.root)
        self.assertEqual(result['steps'], 1000)
        self.assertEqual(len(result['checkpoints_sha256']), 6)
        self.assertEqual(len(result['proofs']), 6)
        (self.root / 'train/seed_2/iter6/cotracker_three_final.pth').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            study.checked_training(self.root)


if __name__ == '__main__':
    unittest.main()
