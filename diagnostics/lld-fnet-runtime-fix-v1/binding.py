"""Independent diagnosis and new-run binding; existing results stay read-only."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'lld-mixed-domain-v1'
sys.path.insert(0, str(OLD))
import protocol as P
import train_domain as BASE
sys.path.insert(0, str(HERE))
from runtime import CONFIG

LIMITS = dict(training_step_seconds=180., warmed_trackrad_seconds=90.,
              maximum_slowdown=1.15, maximum_peak_ratio=.85, maximum_device_fraction=.85,
              output_atol=2e-4, output_rtol=1e-5, grad_atol=2e-6, grad_rtol=1e-3,
              state_atol=2e-6, state_rtol=1e-5)
ARMS = ('F0', 'Fprobe', 'F001', 'F010')
MODES = ('original', 'lowmem')
STEPS = (0, 1, 2)


def sources():
    return {p.relative_to(P.C.CHECKOUT).as_posix(): P.C.sha(p)
            for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.ps1', '.txt')}


def identity(baseline, previous, audit):
    study = P.checked(baseline, previous)
    if study['external_manifest_sha256'] != P.C.sha(audit/'data-manifest.json'):
        raise ValueError('External audit changed')
    return dict(schema=1, baseline_plan_sha256=P.C.sha(baseline/'plan.json'),
                original_sources=P.sources(), repair_sources=sources(), runtime=CONFIG,
                limits=LIMITS, steps=list(STEPS), arms=list(ARMS),
                external_manifest_sha256=study['external_manifest_sha256'])


def checked_manifest(root, baseline, previous, audit):
    current = identity(baseline, previous, audit)
    P.C.freeze(root/'manifest.json', current)
    return current


def job(root, mode, arm):
    if mode not in MODES or arm not in ARMS:
        raise ValueError('Invalid diagnostic job')
    return root/mode/arm


def verified_job(root, mode, arm, deep=True):
    folder = job(root, mode, arm)
    marker = folder/'verified.json'
    if not marker.exists():
        return None
    r = P.C.read(marker)
    if (r.get('complete'), r.get('manifest'), r.get('mode'), r.get('arm')) != (
            True, P.C.sha(root/'manifest.json'), mode, arm):
        raise ValueError('Stale runtime diagnostic')
    P.C.verify(folder, r['files'] if deep else {'steps.json': r['files']['steps.json']})
    return r


def gate(diagnostic, baseline, previous, audit):
    # Read-only verification for production: never create files in /diagnostic.
    if P.C.read(diagnostic/'manifest.json') != identity(baseline, previous, audit):
        raise ValueError('Diagnosis binding changed')
    r = P.C.read(diagnostic/'verified.json')
    if not r.get('complete') or r.get('manifest') != P.C.sha(diagnostic/'manifest.json'):
        raise ValueError('Runtime gate missing')
    P.C.verify(diagnostic, r['files'])
    for arm in ARMS:
        for mode in MODES:
            if not verified_job(diagnostic, mode, arm, deep=False):
                raise ValueError('Incomplete runtime job')
    summary = P.C.read(diagnostic/'summary.json')
    if not summary.get('passed'):
        raise ValueError('Runtime gate closed; inspect diagnostic summary')
    return P.C.sha(diagnostic/'verified.json')


def configure_queue(diagnostic, baseline, previous, audit):
    digest = gate(diagnostic, baseline, previous, audit)
    original_sources = P.sources
    P.sources = lambda: {**original_sources(), **sources()}
    P.RULE = dict(P.RULE, execution_runtime=CONFIG, runtime_diagnosis_sha256=digest,
                  runtime_note='fresh independent run; no old F checkpoints imported')
    return P
