import copy
import ast
import csv
import os
from pathlib import Path
import tempfile
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import common as C
import external as E
import study as D
import train_transfer as TT
import reports as R

C.torch.set_num_threads(1)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        C.save(self.root / 'manifest.json', {'fixture': 1})

    def test_budget_and_seed_process_isolation(self):
        snippets = []
        for seed in C.SEEDS:
            code = """import json, common as C, external as E
from cotracker.utils.teacher_sampling import TeacherPairSampler
s = TeacherPairSampler(3, seed=C.S.STUDENT_SEED)
print(json.dumps({'rule': C.RULE, 'jobs': [E.schedule(i) for i in range(10)],
                  'slots': [s.sample().primary for i in range(10)]}))
"""
            env = dict(os.environ, MCT_FULL_SEED=str(seed))
            raw = subprocess.check_output([sys.executable, '-c', code], cwd=C.PACKAGE, env=env, text=True)
            value = json.loads(raw)
            self.assertEqual(value['rule']['arms'], ['R1', 'B1'])
            self.assertEqual(value['rule']['external_steps'], 3000)
            self.assertEqual(value['rule']['transfer_steps'], 1000)
            self.assertEqual(value['rule']['student_seed'], 20261008 + seed)
            self.assertEqual(value['rule']['data_seed'], 20261023 + seed)
            self.assertEqual(set(value['slots']), {0, 1, 2})
            snippets.append(value['jobs'])
        self.assertEqual(len({json.dumps(jobs) for jobs in snippets}), 3)

    def test_prepare_binds_pilot_but_keeps_it_unchanged(self):
        prior, pilot, ext, audit, weights, out = [self.root / p for p in ('prior', 'pilot', 'ext', 'audit', 'weights', 'out')]
        datasets = []
        ids = {}
        for split, count, prefix in [('train-40', 40, 'T'), ('validation-10', 10, 'V'), ('test-38', 38, 'X')]:
            ids[split] = [f'{prefix}_{i:03}' for i in range(count)]
            for case in ids[split]:
                if split != 'test-38':
                    path = self.root / split / case / 'input.json'
                    C.save(path, {'case': case})
                    checksum = C.digest(path)
                else: checksum = 'a' * 64
                datasets.append({'Split': split, 'Case': case, 'File': f'Z:\\{split}\\{case}\\input.json', 'SHA256': checksum})
        old_prior = {'Source': [], 'Dataset': datasets}
        C.save(prior / 'frozen-study.json', old_prior)
        images = {key: 'sha256:' + 'a' * 64 for key in ('Training', 'Inference', 'Evaluation')}
        C.save(audit / 'data-manifest.json', {})
        C.save(weights / 'fixture.json', {})
        old = {'source_sha256': {}, 'weights_sha256': {'fixture.json': C.digest(weights / 'fixture.json')},
               'data_manifest_sha256': C.digest(audit / 'data-manifest.json'),
               'teacher_pool': C.T.baseline(), 'image': images['Training']}
        C.save(ext / 'frozen-smoke.json', old)
        for name in ('labels',) + C.S.ROUTES:
            C.save(ext / (name + '-verified.json'), {'complete': True, 'plan_sha256': C.digest(ext / 'frozen-smoke.json'), 'files_sha256': {}})
        C.save(ext / 'summary.json', {'complete': True, 'stage': 'external-gpu-smoke', 'steps_per_route': 5,
               'teacher_interfaces': 6, 'routes': list(C.S.ROUTES), 'paired_inputs_and_labels': True,
               'B_fresh_sidecars_exact': True, 'frozen_tensors_and_handoff_passed': True,
               'route_proofs': {r: C.digest(ext / (r + '-verified.json')) for r in C.S.ROUTES}})
        old_pilot = {'source_sha256': {}, 'images': images, 'weights_sha256': old['weights_sha256'],
                     'teacher_pool': C.T.baseline(), 'external_manifest_sha256': old['data_manifest_sha256'],
                     'trackrad_files': {s: C.input_files(old_prior, s) for s in ('train-40', 'validation-10')}}
        C.save(pilot / 'manifest.json', old_pilot)
        selection = pilot / 'reports/validation-10/selection.json'
        C.save(selection, {'eligible_routes': ['B1']})
        C.publish(pilot, 'summary.json', [selection], stage='single-seed-pilot', external_steps=500, transfer_steps=250)
        C.publish(pilot, 'smoke/verified.json', [])
        before = {p.relative_to(pilot).as_posix(): C.digest(p) for p in pilot.rglob('*') if p.is_file()}
        args = (out, prior, audit, ext, weights, self.root / 'train-40', self.root / 'validation-10', images, pilot)
        schedule = {'replacement_steps': 1, 'rows': [{'original_patient': 'patient082', 'slot': 0}]}
        with patch.object(C.S, 'make_jobs'), patch('freeze_sidecar_reference.freeze', side_effect=lambda src, dst: C.save(dst, {'fixture': True})), \
                patch.object(C, 'make_schedule', return_value=schedule), \
                patch.object(C, 'baseline_binding', side_effect=[ValueError('Artifact changed: historical README'), {}, {}]), \
                patch.object(C, 'baseline_steps', return_value=[{'patient': 'patient082', 'slot': 0}]):
            with self.assertRaisesRegex(ValueError, 'historical README'):
                C.prepare(*args)
            self.assertTrue((out / 'exclusion-schedule.json').is_file())
            self.assertFalse((out / 'manifest.json').exists())
            self.assertFalse((out / 'pretrained-frozen.pth').exists())
            C.prepare(*args)
            C.prepare(*args)
        manifest = C.checked_plan(out)
        self.assertEqual(manifest['rule']['external_steps'], 3000)
        self.assertEqual(manifest['trackrad_ids']['test-38'], ids['test-38'])
        self.assertNotIn('test-38', manifest['trackrad_files'])
        self.assertEqual(before, {p.relative_to(pilot).as_posix(): C.digest(p) for p in pilot.rglob('*') if p.is_file()})
        C.save(selection, {'eligible_routes': []})
        with self.assertRaisesRegex(ValueError, 'Artifact changed'): C.prepare(*args)

    def test_three_seed_selection_and_official_aggregation(self):
        fields = {'DSC': .9, 'HD95': 4., 'MASD': 1.5, 'CD': 1.6, 'D98': .95, 'TimeSec': 100.}
        records = [{'Seed': -1, 'Arm': 'pretrained', **fields}]
        for s, delta in enumerate([.001, .0002, -.0001]):
            records += [{'Seed': s, 'Arm': 'R1', **fields},
                        {'Seed': s, 'Arm': 'B1', **dict(fields, DSC=.9 + delta, HD95=3.99)}]
        means, deltas = R.tables(records)
        self.assertEqual([r['N'] for r in means], [1, 3, 3])
        self.assertEqual(len(deltas), 9)
        self.assertTrue(R.choose(deltas)['B1_qualified'])
        self.assertAlmostEqual(means[-1]['DSC_Mean'], .9 + .0011/3)
        self.assertGreater(means[-1]['DSC_SD'], 0)
        bad = copy.deepcopy(deltas)
        for row in bad:
            if row['Reference'] == 'R1': row['Delta_D98'] = -.00201
        self.assertEqual(R.choose(bad)['selected'], 'R1')
        bad = copy.deepcopy(deltas)
        paired = [r for r in bad if r['Reference'] == 'R1']
        for row, delta in zip(paired, [.01, -.0001, -.0001]): row['Delta_DSC'] = delta
        self.assertFalse(R.choose(bad)['B1_qualified'])
        for row, delta in zip(paired, [.01, .01, -.00201]): row['Delta_DSC'] = delta
        self.assertFalse(R.choose(bad)['B1_qualified'])
        with self.assertRaisesRegex(ValueError, 'three paired'): R.choose(deltas[1:])

    def test_plan_seed_mismatch_and_changed_rule_rejected(self):
        plan = {'rule': dict(C.RULE, seed=1, student_seed=20261009, data_seed=20261024),
                'teacher_pool': C.T.baseline(), 'source_sha256': {}}
        C.save(self.root / 'exclusion-schedule.json', {})
        plan['exclusion_schedule_sha256'] = C.digest(self.root / 'exclusion-schedule.json')
        C.save(self.root / 'manifest.json', plan)
        C.save(self.root / 'runtime.json', C.RUNTIME)
        C.checked_plan(self.root, expected_seed=1)
        with self.assertRaisesRegex(ValueError, 'changed'): C.checked_plan(self.root, expected_seed=0)
        plan['rule']['external_steps'] = 500
        C.save(self.root / 'manifest.json', plan)
        with self.assertRaisesRegex(ValueError, 'changed'): C.checked_plan(self.root, expected_seed=1)

    def test_test_gate_requires_valid_frozen_decision(self):
        with patch.object(R, 'training_proofs'):
            with self.assertRaisesRegex(ValueError, 'validation'): R.test_gate(self.root)
            decision = {'test_arms': ['pretrained', 'R1', 'B1'], 'selection_split': 'validation-10'}
            path = self.root / 'reports/validation-10/selection.json'
            C.save(path, decision)
            C.publish(self.root, 'reports/validation-10/verified.json', [path], split='validation-10')
            R.test_gate(self.root)
            C.save(path, dict(decision, test_arms=['B1']))
            with self.assertRaisesRegex(ValueError, 'Artifact changed'): R.test_gate(self.root)

    def test_handoff_rejects_500_step_pilot_as_full(self):
        proof = {'steps': 3000}
        with patch.object(D, 'checked_plan', return_value={'weights_sha256': {}}), \
                patch.object(D, 'marker', side_effect=lambda root, rel: None if '/init/' in rel else proof):
            C.atomic_torch(self.root / 'full/external/B1/external-final.pth', {'route': 'B1', 'steps': 500})
            with self.assertRaisesRegex(ValueError, 'Wrong external handoff'):
                D.handoff(self.root, 'full', 'B1', self.root, self.root)

    def test_report_copies_official_aggregates_and_freezes_selection(self):
        cases = [f'V_{i:03}' for i in range(10)]
        images = {'algorithm': 'fixture'}
        plan = {'trackrad_ids': {'validation-10': cases}, 'student_config': {'border_points': 1000}}
        for seed, arm in [(0, 'pretrained')] + [(s, a) for s in C.SEEDS for a in C.ARMS]:
            local = self.root / f'seed_{seed}'
            C.save(local / 'reference-images.json', images)
            folder = local / 'full/validation-10' / arm
            C.save(folder / 'frozen-images.json', images)
            C.save(folder / 'evaluation-inputs.json', {})
            aggregates = dict(zip(R.FIELDS.values(), [.9 + (.001 if arm == 'B1' else 0), 4., 1.4, 1.5, .95]))
            C.save(folder / C.PROFILE / 'metrics.json', {'aggregates': {**aggregates, 'total_time': 123.},
                   'results': [{'case_id': case, **{f: .5 for f in R.FIELDS.values()}} for case in cases]})
            for case in cases:
                job = folder / C.PROFILE / 'checkpoint/jobs' / case
                output = job / 'output/images/mri-linac-series-targets/output.mha'
                output.parent.mkdir(parents=True)
                output.write_bytes(b'fixture, not a medical prediction')
                (job / '.complete').write_text('fixture')
                C.save(job / 'diagnostics.json', {'profile': C.PROFILE, 'config': plan['student_config'],
                       'model': {'parameters': 25385700 if arm == 'pretrained' else 28763025,
                                 'feature_sidecar': 'none' if arm == 'pretrained' else 'mamba',
                                 'frozen_pretrained': True, 'sidecar_runtime': C.RUNTIME},
                       'prediction': {'output_file_sha256': C.digest(output)},
                       'mechanism': {'sidecar_calls': 2, 'sidecar_residual_rms_max': .001}})
        pairs = [{'Seed': s, 'Candidate': a, **{'Delta_' + k: 0. for k in R.FIELDS}} for s in C.SEEDS for a in ('B1_exclude082','R1_repeat')]
        with patch.object(R, 'training_proofs', return_value=({s: plan for s in C.SEEDS}, [])), patch.object(R, 'eval_bind'), \
                patch.object(R, 'compare_original', return_value=([], pairs)):
            R.report(self.root, 'validation-10')
            R.report(self.root, 'validation-10')
        dest = self.root / 'reports/validation-10'
        with (dest / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[1]['DSC_Mean']), .9)
        self.assertTrue(C.read(dest / 'selection.json')['B1_qualified'])
        C.marker(self.root, 'reports/validation-10/verified.json')

    def test_schedule_balanced_and_restart_independent(self):
        jobs = [E.schedule(i) for i in range(500)]
        self.assertEqual(jobs, [E.schedule(i) for i in range(500)])
        self.assertEqual([E.schedule(i) for i in range(125, 500)], jobs[125:])
        for epoch in range(6):
            self.assertEqual({i for e, i, _ in jobs[epoch*80:(epoch+1)*80]}, set(range(80)))
        self.assertEqual({s for _, _, s in jobs}, {0, 1, 2})

    def test_adapter_compiles_preserves_optimizer_and_loader(self):
        source = (C.APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8')
        adapted = C.instrument(source)
        compile(adapted, 'adapted', 'exec')
        self.assertEqual(adapted.count('__initialize__(model, args)'), 1)
        self.assertEqual(adapted.count('if total_steps >= __stop_steps__:'), 2)
        optimizer = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fetch_optimizer')
        self.assertTrue(ast.get_source_segment(source, optimizer) in adapted)
        for begin, end in [('        resume_epoch = 0', '        enter_training_mode()')]:
            original = source[source.index(begin):]
            original = original[:original.index(end)]
            self.assertTrue(original in adapted)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            C.instrument(source.replace('architecture_report = warm_start(', 'architecture_report = changed('))

    def test_inventory_checks_disjointness_without_test_io(self):
        rows = []
        for split, count, prefix in [('train-40', 40, 'T'), ('validation-10', 10, 'V'), ('test-38', 38, 'X')]:
            for i in range(count):
                case = f'{prefix}_{i:03}'
                rows.append({'Split': split, 'Case': case, 'File': f'Z:\\unavailable\\{case}\\image.mha', 'SHA256': 'a'*64})
        ids = C.validate_inventory({'Dataset': rows})
        self.assertEqual(len(ids['test-38']), 38)
        self.assertEqual(len(C.input_files({'Dataset': rows}, 'train-40')), 40)
        self.assertEqual(len(C.input_files({'Dataset': rows}, 'test-38')), 38)
        rows[-1]['Case'] = 'T_000'
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            C.validate_inventory({'Dataset': rows})

    def test_marker_detects_tampering_and_escape(self):
        path = self.root / 'artifact.json'
        C.save(path, {'x': 1})
        C.publish(self.root, 'proof.json', [path])
        self.assertTrue(C.marker(self.root, 'proof.json')['complete'])
        C.save(path, {'x': 2})
        with self.assertRaisesRegex(ValueError, 'changed'): C.marker(self.root, 'proof.json')
        with self.assertRaises(ValueError): C.verify_files(self.root, {'../escape': 'a'*64})

    def test_extra_training_files_rejected(self):
        C.save(self.root / 'T_001/data.json', {'x': 1})
        files = {'T_001/data.json': C.digest(self.root / 'T_001/data.json'),
                 'manifest.json': C.digest(self.root / 'manifest.json')}
        C.verify_dataset(self.root, files, ['T_001'])
        C.save(self.root / 'T_001/extra.json', {})
        with self.assertRaisesRegex(ValueError, 'file inventory'): C.verify_dataset(self.root, files, ['T_001'])

    def test_external_driver_resume_and_completed_cache(self):
        torch = C.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, video, **kwargs): return video.mean() * self.weight
        class Data:
            def set_epoch(self, epoch): self.epoch = epoch
            def __getitem__(self, index):
                return {'patient': 'patient%03d' % index, 'sampling': {'epoch': self.epoch},
                        'video': torch.ones(2, 3, 2, 2) * (index + 1) / 80}
        def new_model(*args): return Tiny(), {'trainable_names': ['weight']}
        interrupt = [False]
        def label(root, step, sample, weights):
            if interrupt[0] and step == 31: raise RuntimeError('SIMULATED INTERRUPTION')
            path = root / 'full/labels' / f'{step:04}.pth'
            value = {'queries': torch.zeros(1, 1, 3), 'tracks': torch.zeros(1, 2, 1, 2),
                     'visibility': torch.ones(1, 2, 1)}
            if not path.exists():
                C.atomic_torch(path, value); C.publish(root, f'full/labels/{step:04}.json', [path])
            return value, C.digest(path)
        audit = self.root / 'audit'
        C.save(audit / 'data-manifest.json', {})
        plan = {'weights_sha256': {}, 'external_manifest_sha256': C.digest(audit / 'data-manifest.json'), 'baseline_binding': {}}
        roots = [self.root / name for name in ('continuous', 'resumed')]
        for root in roots:
            C.save(root / 'manifest.json', {'fixture': 1})
            C.publish(root, 'smoke/verified.json', [])
            C.publish(root, 'smoke/exclusion-verified.json', [])
        with ExitStack() as stack:
            stack.enter_context(patch.dict(E.RULE, external_steps=50))
            stack.enter_context(patch.object(E, 'gpu'))
            stack.enter_context(patch.object(E, 'checked_plan', return_value=plan))
            stack.enter_context(patch.object(E, 'baseline_binding', return_value={}))
            stack.enter_context(patch.object(E, 'baseline_steps', return_value=[]))
            stack.enter_context(patch.object(E, 'baseline_initial', return_value=C.S.state_hashes(Tiny())))
            stack.enter_context(patch.object(E, 'dataset_class', side_effect=lambda cls: cls))
            stack.enter_context(patch.object(E.S, 'ACDCCineDataset', return_value=Data()))
            stack.enter_context(patch.object(E.S, 'new_model', side_effect=new_model))
            stack.enter_context(patch.object(E, 'label_for', side_effect=label))
            stack.enter_context(patch.object(E.S, 'training_loss', side_effect=lambda result, target: result.square()))
            stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self: self))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self: self))
            for name in ('reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_reserved', return_value=0))
            E.train(roots[0], self.root, audit, self.root, 'B1')
            interrupt[0] = True
            with self.assertRaisesRegex(RuntimeError, 'SIMULATED'): E.train(roots[1], self.root, audit, self.root, 'B1')
            interrupt[0] = False
            E.train(roots[1], self.root, audit, self.root, 'B1')
            with patch.object(E.S, 'new_model', side_effect=AssertionError('Should reuse verified completion')):
                E.train(roots[1], self.root, audit, self.root, 'B1')
        states = [torch.load(r / 'full/external/B1/external-final.pth', weights_only=True)['model'] for r in roots]
        self.assertEqual(C.hashes(states[0]), C.hashes(states[1]))

    def test_resume_rotation_survives_interruption(self):
        plan = C.digest(self.root / 'manifest.json')
        def checkpoint(step):
            return {'route': 'A1', 'steps': step, 'rows': [{'step': i} for i in range(step)],
                    'manifest_sha256': plan, 'model': {'x': C.torch.tensor([step])}}
        E.save_resume(self.root, checkpoint(25))
        E.save_resume(self.root, checkpoint(50))
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 50)
        # Interruption between rotate and publish leaves previous as the sole committed state.
        (self.root / 'resume.pth').rename(self.root / 'uncommitted.pth')
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 25)
        with self.assertRaisesRegex(ValueError, 'Wrong'): E.load_resume(self.root, 'changed', 'A1')
        C.save(self.root / 'resume.pth', {'corrupt': True})
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 25)

    def test_optimizer_resume_equals_uninterrupted(self):
        torch = C.torch
        def model_optimizer():
            C.reset_rng(10)
            model = torch.nn.Linear(4, 2)
            return model, torch.optim.AdamW(model.parameters(), lr=.001)
        def steps(model, optimizer, start, stop):
            for step in range(start, stop):
                C.reset_rng(C.S.STUDENT_SEED + step)
                optimizer.zero_grad(set_to_none=True)
                model(torch.rand(3, 4)).square().mean().backward()
                optimizer.step()
        ref, opt = model_optimizer()
        steps(ref, opt, 0, 10)
        part, optimizer = model_optimizer()
        steps(part, optimizer, 0, 4)
        C.atomic_torch(self.root / 'state.pth', {'model': part.state_dict(), 'optimizer': optimizer.state_dict()})
        resumed, resumed_optimizer = model_optimizer()
        state = torch.load(self.root / 'state.pth', weights_only=True)
        resumed.load_state_dict(state['model']); resumed_optimizer.load_state_dict(state['optimizer'])
        steps(resumed, resumed_optimizer, 4, 10)
        self.assertEqual(C.S.state_hashes(ref), C.S.state_hashes(resumed))

    def test_cached_label_checks_video_and_payload(self):
        video = C.torch.zeros(2, 3, 4, 4)
        sample = {'patient': 'patient001', 'sampling': {'frames': [0, 1]}, 'video': video}
        signature = {'step': 0, 'patient': sample['patient'], 'sampling': sample['sampling'],
                     'video_sha256': C.S.tensor_sha(video[None]), 'slot': E.schedule(0)[2]}
        C.save(self.root / 'exclusion-schedule.json', {'rows': [{'patient': sample['patient'], 'slot': E.schedule(0)[2]}]})
        path = self.root / 'full/labels/0000.pth'
        C.atomic_torch(path, {'signature': signature})
        C.publish(self.root, 'full/labels/0000.json', [path])
        self.assertEqual(E.label_for(self.root, 0, sample, self.root)[0]['signature'], signature)
        sample['video'] = video + 1
        with self.assertRaisesRegex(ValueError, 'clip changed'): E.label_for(self.root, 0, sample, self.root)
        C.atomic_torch(path, {'signature': signature, 'changed': True})
        with self.assertRaisesRegex(ValueError, 'Artifact changed'): E.label_for(self.root, 0, sample, self.root)

    def test_handoff_and_actual_initializer_preserve_adaptation(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, assert_sidecar_training_mode
        torch = C.torch
        torch.manual_seed(8)
        original = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        state = copy.deepcopy(original.state_dict())
        fresh_hashes = []
        for arm in ('R1', 'B1', 'A1'):
            model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            model.load_state_dict(state)
            if arm == 'B1':
                with torch.no_grad(): next(model.fnet.parameters()).add_(.01)
            if arm == 'A1':
                torch.manual_seed(C.S.STUDENT_SEED)
                attach_sidecar(model, 'mamba')
                with torch.no_grad(): model.updateformer.feature_sidecar.output.weight.fill_(.01)
            before = C.S.state_hashes(model)
            proof = C.S.configure_handoff(model, arm)
            after = C.S.state_hashes(model)
            self.assertTrue(all(after[n] == h for n, h in before.items()))
            if arm != 'A1': fresh_hashes.append(proof['sidecar_hashes'])
            path = self.root / 'smoke/init' / (arm + '.pth')
            C.atomic_torch(path, {'model': model.state_dict()})
            C.publish(self.root, f'smoke/init/{arm}.json', [path], state_hashes=after, source_sha256='a'*64)
            loaded = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            report = TT.initialize(loaded, self.root, 'smoke', arm)
            self.assertTrue(report['strict_handoff_exact'])
            self.assertEqual(report['trainable_parameters'], 3377325)
            assert_sidecar_training_mode(loaded)
            self.assertEqual(C.S.state_hashes(loaded), after)
            if arm == 'A1': self.assertTrue(bool((loaded.updateformer.feature_sidecar.output.weight == .01).all()))
            del model, loaded
        self.assertEqual(fresh_hashes[0], fresh_hashes[1])


if __name__ == '__main__': unittest.main()
