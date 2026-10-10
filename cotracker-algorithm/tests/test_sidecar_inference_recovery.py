"""Regression: the real inference entry must not import a stale installed wheel."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'ext' / 'co-tracker'),str(ROOT / 'experiments')]
from verify_sidecar_evaluation_recovery import verify_sources, KNOWN_OLD, ADDED, REVERSE_EDITS
from audit_backbone_growth import file_sha
from local_cotracker_source import activate_local_cotracker


class InferenceRecoveryTests(unittest.TestCase):
    def test_actual_model_entry_with_stale_site_package_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / 'cotracker'
            stale.mkdir()
            (stale / '__init__.py').write_text("raise RuntimeError('STALE INSTALLED COTRACKER')\n")
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join((str(temporary),str(ROOT),env.get('PYTHONPATH','')))
            env.pop('COTRACKER_SOURCE_DIR',None)
            code = "import model; from cotracker.models.core.cotracker import feature_sidecar, cotracker3_offline; print(feature_sidecar.__file__); print(cotracker3_offline.__file__)"
            result = subprocess.run([sys.executable,'-c',code],env=env,cwd=temporary,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(ROOT / 'ext' / 'co-tracker'),result.stdout)
            self.assertNotIn('STALE INSTALLED',result.stderr)

    def test_real_entry_checkpoint_reload_and_cpu_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / 'cotracker'
            stale.mkdir()
            (stale / '__init__.py').write_text("raise RuntimeError('STALE INSTALLED COTRACKER')\n")
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join((str(temporary),str(ROOT),env.get('PYTHONPATH','')))
            env.pop('COTRACKER_SOURCE_DIR',None)
            env.pop('COTRACKER_CHECKPOINT',None)
            code = '''
import model
import torch
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar
torch.set_num_threads(1)
torch.manual_seed(14)
original = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
attach_sidecar(original,'mlp')
original.eval()
with torch.no_grad(): original.updateformer.feature_sidecar.output.weight.normal_(std=.001)
torch.save({'model':original.state_dict(),'feature_sidecar':'mlp','backbone_architecture':'base'},'final.pth')
loaded = model.resources.setup_model('final.pth',device='cpu')
assert loaded.feature_sidecar_kind == 'mlp' and loaded.frozen_pretrained
video = torch.rand(1,3,3,64,64)*255
queries = torch.tensor([[[0.,16.,16.],[1.,32.,32.]]])
with torch.no_grad():
    expected = original(video=video,queries=queries,iters=1)
    actual = loaded(video=video,queries=queries,iters=1)
assert all(torch.equal(a,b) for a,b in zip(expected[:3],actual[:3]))
print('REAL ENTRY CHECKPOINT RELOAD PASSED')
'''
            result = subprocess.run([sys.executable,'-c',code],env=env,cwd=temporary,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CHECKPOINT RELOAD PASSED',result.stdout)

    def test_reject_already_imported_stale_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / 'cotracker'
            stale.mkdir()
            (stale / '__init__.py').write_text('')
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join((str(temporary),str(ROOT),env.get('PYTHONPATH','')))
            result = subprocess.run([sys.executable,'-c',
                'import cotracker; from local_cotracker_source import activate_local_cotracker; activate_local_cotracker()'],
                env=env,cwd=temporary,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('already loaded outside',result.stderr)

    def test_only_exact_reviewed_import_changes_are_reusable(self):
        paths = set(KNOWN_OLD) | ADDED | {'cotracker-algorithm/ext/co-tracker/train_on_real_data.py',
            'cotracker-algorithm/ext/co-tracker/cotracker/models/core/cotracker/feature_sidecar.py',
            'cotracker-algorithm/resources/model.py'}
        current = [{'File':p,'SHA256':file_sha(ROOT.parent / p)} for p in sorted(paths)]
        old = {'Source':[{'File':r['File'],'SHA256':KNOWN_OLD.get(r['File'],r['SHA256'])}
                         for r in current if r['File'] not in ADDED]}
        self.assertEqual(verify_sources(old,current,ROOT.parent),sorted(KNOWN_OLD))
        modified = json.loads(json.dumps(old))
        next(r for r in modified['Source'] if r['File'].endswith('/train_on_real_data.py'))['SHA256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'model/training/protocol'):
            verify_sources(modified,current,ROOT.parent)
        modified = json.loads(json.dumps(old))
        next(r for r in modified['Source'] if r['File']=='cotracker-algorithm/model.py')['SHA256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'unrecognized old'):
            verify_sources(modified,current,ROOT.parent)
        with self.assertRaisesRegex(ValueError,'added/deleted'):
            verify_sources(old,current + [{'File':'scripts/unrelated.py','SHA256':'0'*64}],ROOT.parent)

    def test_allowlisted_file_does_not_allow_extra_numerical_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            current, previous = [], []
            for relative, digest in KNOWN_OLD.items():
                target = repo / relative
                target.parent.mkdir(parents=True,exist_ok=True)
                text = (ROOT.parent / relative).read_text(encoding='utf-8')
                if relative == 'cotracker-algorithm/model.py':
                    text += '\n# An unrelated additional change must still be rejected.\n'
                target.write_text(text,encoding='utf-8')
                current.append({'File':relative,'SHA256':file_sha(target)})
                previous.append({'File':relative,'SHA256':digest})
            with self.assertRaisesRegex(ValueError,'additional numerical changes'):
                verify_sources({'Source':previous},current,repo)

    def test_windows_dockerfile_newlines_are_not_a_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            current, previous = [], []
            for relative, digest in KNOWN_OLD.items():
                target = repo / relative
                target.parent.mkdir(parents=True,exist_ok=True)
                text = (ROOT.parent / relative).read_text(encoding='utf-8')
                if relative.endswith('Dockerfile.inference.reuse'):
                    original = text
                    for new, old in REVERSE_EDITS[relative]: original = original.replace(new,old)
                    digest = hashlib.sha256(original.replace('\n','\r\n').encode()).hexdigest()
                    text = text.replace('\n','\r\n')
                target.write_bytes(text.encode())
                current.append({'File':relative,'SHA256':file_sha(target)})
                previous.append({'File':relative,'SHA256':digest})
            self.assertEqual(verify_sources({'Source':previous},current,repo),sorted(KNOWN_OLD))


if __name__ == '__main__':
    unittest.main()
