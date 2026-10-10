"""Select the copied repository code, not a stale wheel in a reused image."""
import importlib
import os
from pathlib import Path
import sys


def activate_local_cotracker():
    source = Path(os.environ.get('COTRACKER_SOURCE_DIR',
                  Path(__file__).resolve().parent / 'ext' / 'co-tracker')).resolve()
    package = source / 'cotracker'
    if not (package / '__init__.py').is_file():
        raise FileNotFoundError('Local CoTracker package missing: ' + str(package))
    # Never evict loaded modules: that could create mixed old/new class objects.
    for name, module in tuple(sys.modules.items()):
        if name == 'cotracker' or name.startswith('cotracker.'):
            location = getattr(module, '__file__', None)
            if not location or package not in Path(location).resolve().parents:
                raise RuntimeError('CoTracker already loaded outside the selected source: ' + name)
    sys.path[:] = [str(source)] + [p for p in sys.path if p != str(source)]
    importlib.invalidate_caches()
    module = importlib.import_module('cotracker')
    if package not in Path(module.__file__).resolve().parents:
        raise RuntimeError('CoTracker import source mismatch')
    return source
