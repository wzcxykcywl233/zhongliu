"""CPU regression checks: protocol, paired evidence and official report provenance."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
os.environ['COTRACKER_APP_DIR'] = str(REPO / 'cotracker-algorithm')
os.environ['SIDECAR_CHECKOUT'] = str(REPO)
sys.path.insert(0, str(PACKAGE))
import protocol
import study


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sample_rows(policy='old'):
    sampler = study.TeacherPairSampler(3, seed=protocol.BASE_SEED)
    rows = []
    for i in range(protocol.SMOKE_STEPS):
        slot = sampler.sample().primary
        digest = 'b' * 64 if policy != 'old' and slot == 2 else 'a' * 64
        label = {'policy': policy, 'teacher_type': ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot],
                 'query_time_counts': [128] + [0] * 59, 'support_draw_sha256': 'c' * 64,
                 **{k: digest for k in protocol.LABEL_KEYS}, 'teacher_seconds': .3,
                 'visible_gt09_fraction': .9, 'confidence_mean': .99, 'out_of_bounds_fraction': .001,
                 'support_used': protocol.SETTINGS[policy][1] if slot == 2 else None,
                 'iterations': protocol.SETTINGS[policy][2] if slot == 2 else None}
        rows.append({'step': i, 'architecture': 'base', 'teacher_policy': policy, 'primary_index': slot,
                     'auxiliary_index': None, 'case_names': ['A_001'], 'queries_sha256': 'd' * 64,
                     'video_sha256': 'e' * 64, 'label_audit': label, 'loss': .02, 'step_seconds': 2.,
                     'health': {'peak_allocated_mib': 12000., 'peak_reserved_mib': 13000., 'driver_free_mib': 10000.}})
    return rows


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, rows, policy='old'):
        p = self.root / 'rows.jsonl'
        p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        return protocol.read_records(p, protocol.SMOKE_STEPS, policy)

    def test_original_seed_smoke_reaches_real_intervention(self):
        rows = sample_rows()
        self.assertFalse(any(r['primary_index'] == 2 for r in rows[:5]))
        self.assertTrue(any(r['primary_index'] == 2 for r in rows))
        ref = self.parse(rows)
        candidate = self.parse(sample_rows('iter6'), 'iter6')
        self.assertGreater(protocol.compare_rows(ref, candidate), 0)

    def test_replay_accepts_same_inputs_rejects_different_video_or_labels(self):
        rows = sample_rows()
        self.parse(rows + [copy.deepcopy(rows[7])])
        for key in ('video_sha256', 'label_audit'):
            changed = copy.deepcopy(rows[7])
            if key == 'label_audit': changed[key]['tracks_sha256'] = 'f' * 64
            else: changed[key] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'Resumed'):
                self.parse(rows + [changed])

    def test_rejects_old_ten_frame_384_query_recipe(self):
        rows = sample_rows()
        rows[0]['label_audit']['query_time_counts'] = [384] + [0] * 9
        with self.assertRaisesRegex(ValueError, '60-frame'):
            self.parse(rows)

    def test_unchanged_online_labels_and_support_rng_are_paired(self):
        ref = self.parse(sample_rows())
        for key in ('tracks_sha256', 'support_draw_sha256'):
            rows = self.parse(sample_rows('iter6'), 'iter6')
            rows[0]['label_audit'][key] = 'f' * 64
            with self.assertRaises(ValueError): protocol.compare_rows(ref, rows)

    def test_runtime_and_memory_gates(self):
        for field, value in (('peak_reserved_mib', 22000), ('driver_free_mib', 100), ('peak_allocated_mib', float('nan'))):
            rows = sample_rows()
            rows[3]['health'][field] = value
            with self.assertRaises(ValueError): self.parse(rows)
        rows = sample_rows()
        for r in rows: r['step_seconds'] = 25.
        with self.assertRaisesRegex(ValueError, 'slow'): self.parse(rows)

    def test_trainer_adapter_retains_freezing_and_teacher_checks(self):
        path = REPO / 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
        source = path.read_text(encoding='utf-8')
        transformed = protocol.instrument(source)
        compile(transformed, str(path), 'exec')
        self.assertIn('assert_sidecar_training_mode(_unwrap_model(model))', transformed)
        self.assertIn("args.feature_sidecar != 'mamba' or not args.trackrad_data_dir", transformed)
        self.assertIn("Path(args.teacher_offline_ckpt).name != expected", transformed)
        self.assertEqual(transformed.count('if total_steps >= __stop_steps__:'), 2)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            protocol.instrument(source.replace('if total_steps >= args.num_steps:', 'if total_steps > args.num_steps:'))

    def test_smoke_schedule_and_full_initialization_are_separate(self):
        smoke = protocol.train_config(0, 'old', True, 'a')
        full = protocol.train_config(0, 'iter6', False, 'a')
        self.assertEqual((smoke['steps'], full['steps']), (10, 1000))
        self.assertEqual(smoke['scheduler_num_steps'], full['scheduler_num_steps'])
        self.assertEqual(smoke['sidecar_runtime'], protocol.RUNTIME)
        self.assertEqual(full['sequence_len'], 60)

    def fixture_arm(self, folder, kind, score, cases, config, images):
        values = dict(zip(study.FIELDS.values(), (score, 4.2, 1.6, 1.8, .89)))
        write(folder / protocol.PROFILE / 'metrics.json', {'aggregates': {**values, 'total_time': 200.},
              'results': [{'case_id': c, **values} for c in cases]})
        write(folder / 'frozen-images.json', images)
        for case in cases:
            job = folder / protocol.PROFILE / 'checkpoint/jobs' / case
            output = job / 'output/images/mri-linac-series-targets/output.mha'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'fixture-only')
            model = {'parameters': 25385700 if kind == 'none' else protocol.PARAMETERS,
                     'backbone_architecture': 'base', 'frozen_pretrained': True}
            if kind != 'none': model.update(feature_sidecar=kind, sidecar_runtime=protocol.RUNTIME)
            write(job / 'diagnostics.json', {'profile': protocol.PROFILE, 'config': config, 'model': model,
                  'prediction': {'output_file_sha256': protocol.sha(output)},
                  'mechanism': {'sidecar_calls': 2, 'sidecar_residual_rms_max': .01}})
            (job / '.complete').write_text('complete')

    def test_official_report_complete_pairs_negative_results_and_gate_hashes(self):
        cases = ['A_010', 'A_012']
        cfg = {'border_points': 1000}
        images = {'trackrad-algorithm-cotracker-algorithm': 'sha256:' + 'a' * 64}
        write(self.root / 'manifest.json', {'case_ids': {'validation-10': cases}, 'student_config': cfg})
        write(self.root / 'reference-images.json', images)
        folder = self.root / 'full/validation-10'
        self.fixture_arm(folder / 'pretrained', 'none', .9, cases, cfg, images)
        for seed in protocol.SEEDS:
            for i, policy in enumerate(protocol.POLICIES):
                self.fixture_arm(folder / f'seed_{seed}' / policy, 'mamba', .89 + i * .001, cases, cfg, images)
        with patch.object(study, 'checked_training'):
            study.report(self.root, 'validation-10')
            study.test_gate(self.root)
            with (folder / 'frozen-sidecar-teachers-summary.csv').open() as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-1]['N'], '3')
            with (folder / 'frozen-sidecar-teachers-paired-deltas.csv').open() as f: pairs = list(csv.DictReader(f))
            self.assertEqual(len(pairs), 27)
            p = next(p for p in pairs if p['Candidate'] == 'iter6' and p['Reference'] == 'pretrained')
            self.assertAlmostEqual(float(p['Delta_DSC']), -.007)
            path = folder / 'seed_0/old' / protocol.PROFILE / 'checkpoint/jobs/A_010/output/images/mri-linac-series-targets/output.mha'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): study.test_gate(self.root)
            with self.assertRaisesRegex(ValueError, 'output'): study.report(self.root, 'validation-10')

    def test_frozen_nested_config_is_not_overwritten(self):
        p = self.root / 'deep/config.json'
        study.frozen(p, {'a': 1})
        study.frozen(p, {'a': 1})
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            study.frozen(p, {'a': 2})
        self.assertEqual(protocol.read(p), {'a': 1})

    def test_prepare_preserves_originals_and_creates_all_sixteen_arms(self):
        checkout, prior, weights, data, out = (self.root / p for p in ('checkout', 'prior', 'weights', 'datasets', 'out'))
        checkout.mkdir(); prior.mkdir(); weights.mkdir(); out.mkdir()
        trainer = checkout / 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
        trainer.parent.mkdir(parents=True)
        trainer.write_text('# frozen trainer fixture')
        helper = checkout / 'diagnostics/frozen-teacher-full-v1/study.py'
        helper.parent.mkdir(parents=True)
        helper.write_text('# frozen helper fixture')
        inventory = []
        for split, alias, prefix, count in (('train-40', 'train', 'A', 40), ('validation-10', 'validation', 'B', 10), ('test-38', 'test', 'C', 38)):
            for i in range(count):
                case = f'{prefix}_{i:03}'
                f = data / alias / case / 'frames.mha'
                f.parent.mkdir(parents=True)
                f.write_bytes(b'fixture-not-medical-data')
                inventory.append({'Split': split, 'Case': case, 'File': f'C:\\data\\{alias}\\{case}\\frames.mha', 'SHA256': protocol.sha(f)})
        weights_inventory = []
        for name in ('scaled_offline.pth', 'baseline_offline.pth', 'baseline_online.pth', 'cotracker2v1.pth'):
            study.torch.save({'weight': study.torch.ones(2)}, weights / name)
            weights_inventory.append({'Name': name, 'SHA256': protocol.sha(weights / name)})
        write(prior / 'frozen-study.json', {'Source': [{'File': trainer.relative_to(checkout).as_posix(), 'SHA256': protocol.sha(trainer)}],
              'Weights': weights_inventory, 'Dataset': inventory})
        images = {k: 'sha256:' + d * 64 for k, d in (('Training', 'a'), ('Inference', 'b'), ('Evaluation', 'c'))}
        write(prior / 'local-runtime-images.json', images)
        write(prior / 'reference-images.json', {'trackrad-algorithm-cotracker-algorithm': images['Inference'], 'trackrad-evaluation': images['Evaluation']})
        before = {str(p): protocol.sha(p) for p in prior.iterdir()}
        # The existing Linux-only checkpoint exporter fsyncs a read handle,
        # which Windows rejects. File contents/hash checks remain real here.
        with patch.object(study, 'CHECKOUT', checkout), patch('freeze_sidecar_reference.os.fsync'):
            study.prepare(out, prior, PACKAGE, weights, data)
            study.prepare(out, prior, PACKAGE, weights, data)
            self.assertEqual(len(list(out.glob('*/train/seed_*/*/run-config.json'))), 16)
            self.assertEqual(protocol.read(out / 'manifest.json')['policies'], list(protocol.POLICIES))
            self.assertEqual(before, {str(p): protocol.sha(p) for p in prior.iterdir()})
            trainer.write_text('# incompatible source')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                study.prepare(out, prior, PACKAGE, weights, data)

    def test_real_model_checkpoint_audit_rejects_frozen_backbone_change(self):
        study.torch.set_num_threads(1)
        model = study.CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        info = study.attach_sidecar(model, 'mamba')
        info.update(execution_runtime=protocol.RUNTIME, initialization_outputs_exact=True,
                    initialization_reference='frozen-pretrained', initialization_max_abs_difference=[0., 0., 0.])
        with study.torch.no_grad(): model.updateformer.feature_sidecar.output.weight.fill_(.001)
        manifest = {'trainer_sha256': 'a' * 64, 'package_sha256': {'train.py': 'b' * 64, 'protocol.py': 'c' * 64}}
        write(self.root / 'manifest.json', manifest)
        write(self.root / 'reference-proof.json', {'base_tensor_sha256': info['base_tensor_sha256']})
        folder = self.root / 'smoke/train/seed_0/old'
        write(folder / 'run-config.json', protocol.train_config(0, 'old', True, protocol.sha(self.root / 'manifest.json')))
        write(folder / 'initialization-report.json', {'architecture': 'base', 'initialization_outputs_exact': True, 'sidecar': info})
        write(folder / 'trainer-adapter.json', {'source_sha256': 'a' * 64, 'adapter_sha256': 'b' * 64,
              'protocol_sha256': 'c' * 64, 'stop_steps': protocol.SMOKE_STEPS, 'scheduler_steps': 1000})
        (folder / 'paired-steps.jsonl').write_text('\n'.join(json.dumps(r) for r in sample_rows()))
        checkpoint = {'total_steps': protocol.SMOKE_STEPS, 'feature_sidecar': 'mamba',
                      'backbone_architecture': 'base', 'teacher_student_policy': 'old', 'model': model.state_dict()}
        with patch.object(study.torch, 'load', return_value=checkpoint):
            rows, _ = study.verify_arm(self.root, 'smoke', 0, 'old')
            self.assertEqual(len(rows), protocol.SMOKE_STEPS)
            key = next(k for k in checkpoint['model'] if '.feature_sidecar.' not in k)
            checkpoint['model'][key] = checkpoint['model'][key] + .01
            with self.assertRaisesRegex(ValueError, 'backbone tensors changed'):
                study.verify_arm(self.root, 'smoke', 0, 'old')


if __name__ == '__main__':
    unittest.main()
