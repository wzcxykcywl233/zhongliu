"""The local CoTracker setup backend must not require a PyPI bootstrap."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LocalPackageBuildTests(unittest.TestCase):
    def test_only_local_cotracker_uses_installed_locked_backend(self):
        manifest = tomllib.loads((ROOT/'pixi.toml').read_text())
        self.assertEqual(manifest['pypi-options']['no-build-isolation'], ['cotracker'])
        self.assertEqual(manifest['dependencies']['setuptools'], '==80.9.0')
        lock = (ROOT/'pixi.lock').read_text()
        self.assertIn('noarch/setuptools-80.9.0-pyhff2d567_0.conda',lock)
        self.assertNotIn('tls-no-verify',manifest['pypi-options'])

    @unittest.skipUnless(importlib.util.find_spec('setuptools'), 'setuptools unavailable')
    def test_actual_setup_builds_wheel_without_installer_or_network(self):
        # Copy only packaging inputs; never generate egg-info in the worktree.
        # setup.py is pure Python and has no mandatory install_requires.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shutil.copy2(ROOT/'ext/co-tracker/setup.py',path/'setup.py')
            shutil.copytree(ROOT/'ext/co-tracker/cotracker',path/'cotracker',
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            code = (
                "import socket; "
                "socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden')); "
                "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network forbidden')); "
                "import setuptools.build_meta as backend; "
                "backend.build_wheel('dist')"
            )
            result = subprocess.run([sys.executable,'-c',code],cwd=path,
                                    capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertEqual(len(list((path/'dist').glob('cotracker-3.0-*.whl'))),1)


if __name__ == '__main__':
    unittest.main()
