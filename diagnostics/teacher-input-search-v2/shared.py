"""Read-only reuse of the previously audited frozen-student implementation."""
import importlib.util
import os
from pathlib import Path
import sys

CHECKOUT = Path(os.environ.get('SIDECAR_CHECKOUT', '/checkout'))
APP = Path(os.environ.get('COTRACKER_APP_DIR', '/opt/app'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


directory = CHECKOUT / 'diagnostics/frozen-sidecar-teachers-v1'
legacy_protocol = load('original_frozen_protocol', directory / 'protocol.py')
previous = sys.modules.get('protocol')
try:
    sys.modules['protocol'] = legacy_protocol
    legacy = load('original_frozen_study', directory / 'study.py')
finally:
    if previous is None:
        sys.modules.pop('protocol', None)
    else:
        sys.modules['protocol'] = previous
