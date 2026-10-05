import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class BackboneRuntimeReuseTests(unittest.TestCase):
    def test_dependency_check_records_versions_without_claiming_gpu_training(self):
        spec = importlib.util.spec_from_file_location('backbone_runtime_check', ROOT / 'experiments/check_backbone_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fake_trainer = types.SimpleNamespace(fetch_optimizer=lambda: None, __file__='/opt/app/ext/co-tracker/train_on_real_data.py')
        with tempfile.TemporaryDirectory() as temp, patch.dict('sys.modules', {'train_on_real_data': fake_trainer}), patch.object(module.importlib.metadata, 'version', return_value='1.9.4'):
            output = Path(temp) / 'runtime.json'
            report = module.check(output)
            self.assertTrue(output.is_file())
            self.assertEqual(report['architectures'], ['base','time6','space_time6'])
            self.assertEqual(report['pytorch_lightning'], '1.9.4')
            self.assertIn('CPU dependency check only', report['note'])

    def test_reuse_build_copies_current_code_without_any_installer(self):
        dockerfile = (ROOT / 'Dockerfile.training.reuse').read_text()
        self.assertNotIn('\nRUN ', dockerfile)
        self.assertIn('COPY --chown=user:user ext/co-tracker/', dockerfile)
        self.assertIn('COPY --chown=user:user experiments/', dockerfile)
        runner = (ROOT.parent / 'scripts/run_backbone_growth_resumable.ps1').read_text()
        self.assertIn("'--file', $Dockerfile", runner)
        self.assertIn("'--pull=false','--network=none'", runner)
        self.assertIn("'tag',$RuntimeId,$RuntimeAlias", runner)
        self.assertIn('TrainingRuntimeImage=$TrainingRuntimeImage', runner)
        self.assertIn("'check_backbone_runtime.py'", runner)


if __name__ == '__main__':
    unittest.main()
