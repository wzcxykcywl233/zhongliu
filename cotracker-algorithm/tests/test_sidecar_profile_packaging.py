"""Emulate the real reuse Dockerfile COPY closure; do not substitute a loader."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ProfilePackagingTests(unittest.TestCase):
    def test_reused_training_image_updates_loader_and_reloads_three_branches(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary)/'app'
            (stage/'resources').mkdir(parents=True)
            # This stands for the stale resources loader inherited from the
            # old training image. The actual Dockerfile must replace it.
            (stage/'resources/__init__.py').write_text('')
            (stage/'resources/model.py').write_text("raise RuntimeError('STALE INHERITED LOADER')\n")
            copied=[]
            dockerfile=(ROOT/'Dockerfile.training.reuse').read_text(encoding='utf-8')
            self.assertIn('ENV PYTHONPATH=/opt/app/ext/co-tracker:/opt/app',dockerfile)
            for line in dockerfile.splitlines():
                if not line.startswith('COPY '): continue
                _,_,source,destination=shlex.split(line)
                self.assertTrue(destination.startswith('/opt/app/'))
                target=stage/destination[len('/opt/app/'):]
                for path in ROOT.glob(source.rstrip('/')):
                    if path.is_dir(): shutil.copytree(path,target,dirs_exist_ok=True)
                    else:
                        target.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(path,target/path.name)
                copied.append(source.rstrip('/'))
            self.assertIn('resources',copied)
            self.assertIn('*.py',copied)
            stale=Path(temporary)/'site-packages/cotracker'
            stale.mkdir(parents=True)
            (stale/'__init__.py').write_text("raise RuntimeError('STALE INSTALLED WHEEL')\n")
            env=dict(os.environ)
            env['PYTHONPATH']=os.pathsep.join((str(stale.parent),str(stage),str(stage/'experiments'),env.get('PYTHONPATH','')))
            for key in ('COTRACKER_SOURCE_DIR','COTRACKER_CHECKPOINT','TRACKRAD_SIDECAR_RUNTIME'):
                env.pop(key,None)
            code='''
from pathlib import Path
import torch
import profile_sidecar_speed as profile
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar
torch.set_num_threads(1)
loader_sha=profile.file_sha(Path(profile.checkpoint_loader.__file__))
assert Path(profile.checkpoint_loader.__file__).resolve()==Path('resources/model.py').resolve()
for kind in ('mlp','mamba_short','mamba'):
    torch.manual_seed(11)
    original=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
    attach_sidecar(original,kind)
    original.eval()
    with torch.no_grad(): original.updateformer.feature_sidecar.output.weight.normal_(std=.001)
    torch.save({'model':original.state_dict(),'feature_sidecar':kind,'backbone_architecture':'base'},'fixture.pth')
    loaded=profile.load_profile_model(Path('fixture.pth'),kind,'cpu',loader_sha)
    assert loaded.state_dict().keys()==original.state_dict().keys()
    assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in original.state_dict().items())
    video=torch.rand(1,3,3,64,64)*255
    queries=torch.tensor([[[0.,16.,16.],[1.,32.,32.]]])
    with torch.no_grad():
        expected=original(video=video,queries=queries,iters=1)
        actual=loaded(video=video,queries=queries,iters=1)
    assert all(torch.equal(a,b) for a,b in zip(expected[:3],actual[:3]))
    print('PACKAGED STRICT RELOAD PASSED '+kind,flush=True)
    del original,loaded
try:
    profile.load_profile_model(Path('fixture.pth'),'mamba','cpu','0'*64)
except ValueError as error:
    assert 'Stale checkpoint loader' in str(error)
else: raise AssertionError('Changed loader accepted')
'''
            result=subprocess.run([sys.executable,'-c',code],env=env,cwd=stage,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.count('PACKAGED STRICT RELOAD PASSED'),3)


if __name__=='__main__': unittest.main()
