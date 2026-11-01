"""Independent 100-update preflight; does not replace the strict numerical gate."""
from pathlib import Path
import math
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'lld-same-state-verification-v1'))
import verify as V
from domain import domain_loss, gradient_norm
sys.path.insert(0, str(HERE))
B, P, C, T, S = V.B, V.P, V.C, V.T, V.S
ARMS = V.ARMS
RULE = dict(schema=1, seed=0, arms=list(ARMS), updates=100,
    fixed_probe_sample_indices=[100, 101, 102, 103], probe_after_updates=[0, 25, 50, 75, 100],
    checkpoint_every=10, audit_steps=[0, 1, 24, 49, 74, 99],
    maximum_step_seconds=180., maximum_device_fraction=.85,
    probe_loss_review_ratio=2., probe_loss_review_absolute_increase=.01,
    maximum_gpu_job_seconds=7200, previous_strict_gate_changed=False,
    full_training_authorized=False,
    note='User-authorized bounded stability check, NOT numerical equivalence or task-quality acceptance. '
         'Loss-growth alerts are exploratory review flags, not accuracy conclusions. '
         'F arms train Mamba plus fnet.layer4/conv2/conv3; remaining backbone tensors stay exact. '
         'No pilot/full/evaluation queue follows automatically.')


def sources():
    return {p.relative_to(C.CHECKOUT).as_posix(): C.sha(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.ps1', '.txt')}


def evidence_binding(evidence, baseline, previous, audit, deep=False):
    expected = dict(rule=V.RULE, upstream=B.identity(baseline, previous, audit), sources=V.sources())
    if C.read(evidence/'manifest.json') != expected: raise ValueError('Same-state source binding changed')
    summary = C.read(evidence/'reports/all/summary.json')
    if not summary['complete'] or summary['arms'] != list(ARMS) or summary['gate_changed']:
        raise ValueError('Completed unchanged four-arm same-state diagnosis required')
    if summary['manifest_sha256'] != C.sha(evidence/'manifest.json'):
        raise ValueError('Diagnosis manifest changed')
    files = ['manifest.json', 'environment.json', 'reports/all/summary.json']
    for arm in ARMS:
        for job in V.JOBS:
            f = V.folder(evidence, arm, job)
            r = C.read(f/'complete.json')
            if (r['complete'], r['manifest'], r['environment_sha256'], r['arm'], r['job']) != (
                    True, C.sha(evidence/'manifest.json'), C.sha(evidence/'environment.json'), arm, job):
                raise ValueError('Stale numerical evidence receipt')
            if C.sha(f/'complete.json') != summary['job_receipts'][arm+'/'+job]:
                raise ValueError('Numerical evidence receipt changed')
            C.verify(f, {'steps.json': r['files']['steps.json']})
            if deep and not V.receipt(evidence, arm, job): raise ValueError('Missing numerical evidence')
            files += [f'{arm}/{job}/complete.json', f'{arm}/{job}/steps.json']
    return {n: C.sha(evidence/n) for n in files}


def plan(root, evidence, baseline, previous, audit, deep=False):
    value = dict(rule=RULE, sources=sources(),
                 evidence=evidence_binding(evidence, baseline, previous, audit, deep),
                 upstream=B.identity(baseline, previous, audit))
    C.freeze(root/'plan.json', value)
    return value


def check_environment(root, evidence):
    current = V.environment()
    if current != C.read(evidence/'environment.json'):
        raise ValueError('GPU/runtime differs from prior diagnostic; use a separately reviewed run')
    C.freeze(root/'environment.json', current)


def load(path): return torch.load(path, map_location='cpu', weights_only=True)


def publish(root, folder, names, kind):
    C.save(folder/'complete.json', dict(complete=True, kind=kind, plan_sha256=C.sha(root/'plan.json'),
        environment_sha256=C.sha(root/'environment.json'), files={n: C.sha(folder/n) for n in names}))


def completed(root, folder, kind):
    path = folder/'complete.json'
    if not path.exists(): return False
    r = C.read(path)
    if (r['complete'], r['kind'], r['plan_sha256'], r['environment_sha256']) != (
            True, kind, C.sha(root/'plan.json'), C.sha(root/'environment.json')):
        raise ValueError('Stale stability receipt')
    expected = {'summary.json'} if kind == 'graph' else {'resume.pth', 'steps.json', 'probes.json', 'proof.json'}
    if set(r['files']) != expected: raise ValueError('Incomplete stability receipt')
    C.verify(folder, r['files'])
    return True


def validate_records(rows):
    if [r['step'] for r in rows] != list(range(len(rows))) or len(rows) > RULE['updates']:
        raise ValueError('Broken update history')
    for r in rows:
        for k in ('point', 'domain_loss', 'student_grad_norm', 'classifier_grad_norm', 'grl_strength'):
            if not math.isfinite(r[k]) or r[k] < 0: raise ValueError('Invalid training quantity: '+k)
        if not 0 < r['seconds'] <= RULE['maximum_step_seconds']:
            raise ValueError('Step runtime exceeded 180s or invalid; do not skip on resume')
        if not math.isfinite(r['device_total']) or not 0 < r['peak_allocated'] <= RULE['maximum_device_fraction']*r['device_total']:
            raise ValueError('Insufficient GPU memory margin; do not skip on resume')


def loss_alert(initial, final):
    if any(not math.isfinite(v) or v < 0 for v in (initial, final)):
        raise ValueError('Invalid fixed-probe loss')
    return final > RULE['probe_loss_review_ratio']*initial and final-initial > RULE['probe_loss_review_absolute_increase']


def validate_probes(probes, completed_updates):
    expected = {(n, i) for n in RULE['probe_after_updates'] if n <= completed_updates
                for i in RULE['fixed_probe_sample_indices']}
    if len(probes) != len(expected) or {(p['after_updates'], p['sample_index']) for p in probes} != expected:
        raise ValueError('Incomplete or duplicate fixed probes')
    for p in probes:
        for k in ('point_loss', 'coordinate_shift_rms_px', 'visibility_mean_abs_change', 'confidence_mean_abs_change'):
            if not math.isfinite(p[k]) or p[k] < 0: raise ValueError('Invalid fixed probe: '+k)
        for k in ('visibility_mean', 'confidence_mean', 'out_of_bounds_fraction'):
            if not math.isfinite(p[k]) or not 0 <= p[k] <= 1: raise ValueError('Invalid fixed probe: '+k)


def graph_check(model, head, embedding, task, domain):
    """Test the actual hooked fnet graph; autograd.grad never updates weights."""
    params = [p for p in model.parameters() if p.requires_grad]
    enc = [p for n, p in model.named_parameters() if n.startswith(P.ENCODER_PREFIXES) and p.requires_grad]
    loss0, _ = domain_loss(head, embedding, domain, 0.)
    zero = torch.autograd.grad(loss0, params+[embedding], retain_graph=True, allow_unused=True)
    if any(g is not None for g in zero): raise ValueError('Zero GRL leaks a graph path to the student')
    hgrads = torch.autograd.grad(loss0, list(head.parameters()), retain_graph=True)
    if not all(torch.isfinite(g).all() for g in hgrads) or not any(torch.count_nonzero(g) for g in hgrads):
        raise ValueError('Detached classifier cannot learn')
    positive, _ = domain_loss(head, embedding, domain, .01)
    adv = gradient_norm(positive, enc)
    task_norm = gradient_norm(task, enc)
    if not math.isfinite(adv) or not math.isfinite(task_norm) or min(adv, task_norm) <= 0:
        raise ValueError('Positive GRL/task encoder gradients missing')
    return dict(zero_student_gradient_paths=0, zero_embedding_gradient_is_none=True,
                checked_student_parameters=len(params), classifier_gradient_nonzero=True,
                positive_grl_encoder_grad_norm=adv, task_encoder_grad_norm=task_norm,
                positive_strength=.01, positive_strength_is_diagnostic_only=True)
