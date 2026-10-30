"""Read-only binding to rs-v1; explicit post-gate G1 exploration, not a gate repair."""
from pathlib import Path
import importlib.util
import sys

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE.parent / 'region-supervision-v1'
sys.path.insert(0, str(ORIGINAL))
import core as C

RULE = dict(schema=1, seeds=[0, 1, 2], g1_steps=2000,
    g1_authorization='User explicitly requested G1 after the failed D1-region gate',
    g1_gate_override=True, g1_split='validation-10', g1_test=False,
    g1_change='Remove teacher coordinate loss only; keep weak boundary/region losses and visibility gates',
    point_test=['pretrained', 'M0', 'W1', 'D1'],
    diagnostic_split='validation-10', diagnostic_modes=['raw', 'percentile', 'prior', 'static'],
    diagnostic_selection=False, diagnostic_threshold=0.,
    note='Point test is a requested subset, not completion of the old whole-output test plan. '
         'G1 is post-gate exploratory. Public test-38 already inspected; no blind-test claim.')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def source_files():
    folders = [PACKAGE, PACKAGE.parent/'region-output-shape-fix-v1',
               PACKAGE.parent/'region-short-teacher-fix-v1']
    return {p.relative_to(C.CHECKOUT).as_posix(): C.sha(p) for d in folders
            for p in sorted(d.iterdir()) if p.suffix in ('.py', '.ps1', '.txt')}


def training_receipt(previous, arm, seed, phase='full'):
    folder = previous/phase/f'train/seed_{seed}'/arm
    r = C.cached(previous, folder/'verified.json')
    steps = C.RULE[phase+'_steps']
    if not r or (r['arm'], r['seed'], r['phase'], r['steps']) != (arm, seed, phase, steps):
        raise ValueError('Missing/wrong completed training: '+str(folder))
    info = C.read(folder/'proof.json')
    rows = C.read(folder/'steps.json')
    if not r.get('reload_gpu_forward') or not info['frozen_backbone_exact']:
        raise ValueError('Missing frozen/GPU proof')
    if [x['step'] for x in rows] != list(range(steps)):
        raise ValueError('Incomplete step history')
    return folder, rows, info


def prepare(root, previous):
    original = C.checked(previous)
    report = previous/'reports/full/validation-10'
    required = [previous/'smoke/audit.json', previous/'reports/pilot/validation-10/verified.json',
                report/'verified.json']
    for path in required:
        if not C.cached(previous, path):
            raise ValueError('Missing original prerequisite: '+str(path))
    decision = C.read(report/'selection.json')
    if decision['selection_split'] != 'validation-10':
        raise ValueError('Selection must precede test')
    expected = [['M0','points'], ['W1','points'], ['D1','points'], ['D1','region']]
    if decision['test_outputs'] != expected:
        raise ValueError('Unexpected original frozen test plan')
    required += [previous/'manifest.json', report/'selection.json']
    # Freeze inputs, not merely a checkpoint name. Old files are never rewritten.
    for seed in RULE['seeds']:
        for arm in ('M0', 'W1', 'D1'):
            folder, _, _ = training_receipt(previous, arm, seed)
            required += [folder/n for n in ('verified.json', 'proof.json', 'steps.json', 'final.pth')]
        for arm, view in expected:
            folder = previous/f'full/validation-10/seed_{seed}'/arm/view
            if not C.cached(previous, folder/'metrics-verified.json'):
                raise ValueError('Missing original official metrics')
            required += [folder/'metrics.json', folder/'metrics-verified.json']
    training_receipt(previous, 'G1', 0, 'smoke')
    required += [previous/'smoke/train/seed_0/G1/verified.json']
    value = dict(rule=RULE, sources=source_files(), original_region_gate=decision['region_gate'],
                 previous_files={p.relative_to(previous).as_posix():C.sha(p) for p in required},
                 ids=original['ids'], images=original['images'])
    C.freeze(root/'followup-plan.json', value)
    # Exact old manifest permits reusing its immutable trainer without changing its sources.
    g1 = root/'g1'
    g1.mkdir(parents=True, exist_ok=True)
    dest = g1/'manifest.json'
    if dest.exists() and C.sha(dest) != C.sha(previous/'manifest.json'):
        raise ValueError('G1 compatibility manifest changed')
    if not dest.exists(): dest.write_bytes((previous/'manifest.json').read_bytes())
    C.freeze(g1/'authorization.json', dict(followup_plan_sha256=C.sha(root/'followup-plan.json'),
        explicit_override=True, original_region_gate=decision['region_gate'],
        scope='full/G1 only; validation-10 only; original selection preserved'))
    print('PLAN VERIFIED: point test + read-only D1 diagnosis + explicitly authorized G1 exploration', flush=True)


def checked(root, previous):
    p = C.read(root/'followup-plan.json')
    if p['rule'] != RULE or p['sources'] != source_files():
        raise ValueError('Follow-up sources/plan changed; use new output root')
    C.verify(previous, p['previous_files'])
    C.checked(previous)
    return p


def receipt(root, path, files, **extra):
    C.save(path, dict(complete=True, plan=C.sha(root/'followup-plan.json'),
        files={p.relative_to(root).as_posix():C.sha(p) for p in files}, **extra))


def cached(root, path):
    if not path.exists(): return None
    r = C.read(path)
    if not r.get('complete') or r['plan'] != C.sha(root/'followup-plan.json'):
        raise ValueError('Stale follow-up receipt')
    C.verify(root, r['files'])
    return r


def authorize_g1(root, previous, target, phase, arm):
    checked(root, previous)
    a = C.read(root/'g1/authorization.json')
    if (target.resolve(), phase, arm) != ((root/'g1').resolve(), 'full', 'G1') or not a['explicit_override']:
        raise ValueError('Override permits full/G1 in the isolated output only')
    if a['followup_plan_sha256'] != C.sha(root/'followup-plan.json'):
        raise ValueError('Authorization binding changed')


PAIR_KEYS = ('step','video','queries','sampling','domain','slot','draw',
             'label_tracks','label_visibility','label_confidence')


def compare_g1(rows, other, proof, refproof):
    if len(rows) != RULE['g1_steps'] or [r['step'] for r in rows] != list(range(RULE['g1_steps'])):
        raise ValueError('Incomplete G1 steps')
    if [{k:r[k] for k in PAIR_KEYS} for r in rows] != [{k:r[k] for k in PAIR_KEYS} for r in other]:
        raise ValueError('G1/D1 inputs or labels differ')
    if any(r['point'] != 0 for r in rows): raise ValueError('G1 still has point loss')
    if not proof['frozen_backbone_exact'] or any(proof[k] != refproof[k] for k in ('initial_sidecar','initial_head','domain_counts')):
        raise ValueError('Unpaired initialization/frozen backbone')
