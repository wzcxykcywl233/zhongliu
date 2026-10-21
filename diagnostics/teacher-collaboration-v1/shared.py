"""Reuse frozen infrastructure without modifying previous experiments."""
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


legacy_protocol = load('frozen_protocol', CHECKOUT / 'diagnostics/frozen-sidecar-teachers-v1/protocol.py')
previous = sys.modules.get('protocol')
try:
    sys.modules['protocol'] = legacy_protocol
    legacy = load('frozen_study', CHECKOUT / 'diagnostics/frozen-sidecar-teachers-v1/study.py')
finally:
    if previous is None:
        sys.modules.pop('protocol', None)
    else:
        sys.modules['protocol'] = previous
base_protocol = load('factorial_protocol', CHECKOUT / 'diagnostics/teacher-combination-search-v4/protocol.py')
