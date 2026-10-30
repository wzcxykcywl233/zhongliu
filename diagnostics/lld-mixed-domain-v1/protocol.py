"""LLD synthetic mixed study; old ACDC studies are immutable read-only references."""
from pathlib import Path
import importlib.util
import sys

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE.parent / 'region-supervision-v1'
sys.path.insert(0, str(ORIGINAL))
import core as C

# (representation, gradient-reversal maximum). Zero arms are detached probes.
ARMS = {'M0': ('none', 0.), 'S0': ('sidecar', 0.),
        'S001': ('sidecar', .001), 'S010': ('sidecar', .01),
        'F0': ('none', 0.), 'Fprobe': ('fnet', 0.),
        'F001': ('fnet', .001), 'F010': ('fnet', .01)}
FAMILIES = {'sidecar': ['S001', 'S010'], 'fnet': ['F001', 'F010']}
PREFIX = 'updateformer.feature_sidecar.'
ENCODER_PREFIXES = ('fnet.layer4.', 'fnet.conv2.', 'fnet.conv3.')
PAIR_KEYS = ('step', 'video', 'queries', 'sampling', 'domain', 'slot', 'draw',
             'label_tracks', 'label_visibility', 'label_confidence')
RULE = dict(schema=1, arms={k:list(v) for k,v in ARMS.items()},
    smoke_steps=6, pilot_steps=250, full_steps=2000, seeds=[0,1,2],
    encoder_prefixes=list(ENCODER_PREFIXES), encoder_lr=5e-6, sidecar_lr=5e-5,
    classifier_lr=1e-4, weight_decay=1e-5, warmup_steps=250,
    domain_head='LayerNorm(C), Linear(C,128), GELU, Linear(128,1); no dropout/BatchNorm',
    pool='mean non-query frames and spatial/point tokens; one balanced-domain clip per alternating step',
    fnet_tap='raw final fnet output, all chunks, before tracker L2 normalization',
    sidecar_tap='last block latent, last refinement iteration, detached backbone inputs unchanged',
    classifier_clip=10., student_clip=10., independent_optimizers=True,
    initialization='official scaled + fresh compact Mamba, not B1; same TrackRAD even-step sample/label sequence; LLD synthetic odd steps',
    task='TrackRAD cached teacher coordinates + exact synthetic LLD coordinates; no region head',
    selection='per-family best qualifying validation DSC; D98>=reference-.002, HD95<=reference+.02; '
              'F must also qualify vs M0; full >=2/3 DSC positive, worst>=-.002',
    test='explicit only; fixed full arms; never select from test',
    note='LLD patient-group 80/20, synthetic time only; TrackRAD40/10/38. Not an ACDC dataset-only comparison: motion/labels also differ. Domain confusion is not proof of accuracy '
         'or invariance. No clinical/independent blind-test claim.')
from synthetic import MOTION
RULE.update(external='LLD-MMRI-MedSAM2', synthetic_motion=MOTION, mixed_updates='TrackRAD/LLD 1:1',
            supervision='Teacher pseudo-labels on TrackRAD, geometric targets on LLD; no real LLD video')
TRACKRAD_PAIR_KEYS=PAIR_KEYS
PAIR_KEYS=PAIR_KEYS+('label_origin',)
sys.path.insert(0,str(PACKAGE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def sources():
    folders = [PACKAGE, PACKAGE.parent/'region-output-shape-fix-v1',
               PACKAGE.parent/'region-short-teacher-fix-v1']
    return {p.relative_to(C.CHECKOUT).as_posix():C.sha(p) for d in folders
            for p in sorted(d.iterdir()) if p.suffix in ('.py','.ps1','.txt')}


def old_training(previous, seed):
    folder = previous/f'full/train/seed_{seed}/M0'
    proof = C.cached(previous, folder/'verified.json')
    if not proof or (proof['phase'], proof['arm'], proof['seed'], proof['steps']) != ('full','M0',seed,2000):
        raise ValueError('Completed rs-v1/M0 required')
    info = C.read(folder/'proof.json'); rows = C.read(folder/'steps.json')
    if not info['frozen_backbone_exact'] or not proof.get('reload_gpu_forward'):
        raise ValueError('Original frozen/GPU proof missing')
    if [r['step'] for r in rows] != list(range(2000)):
        raise ValueError('Old sequence incomplete')
    return folder, rows, info


def prepare(root, previous, audit):
    import dataset as D
    ext=C.read(audit/'data-manifest.json')
    if not ext.get('training_ready') or ext['revision']!=D.REVISION or ext['audit_code']!=C.sha(PACKAGE/'dataset.py'):
        raise ValueError('Completed current LLD audit required')
    if len(ext['patients'])!=498 or len(ext['scans'])!=3984: raise ValueError('Unexpected external cohort')
    for row in ext['scans']:
        if row['split']!=ext['patients'][row['patient']] or not row['usable_slices']: raise ValueError('Bad external assignment')
    for group in ext['exact_duplicate_patient_groups']:
        if len({ext['patients'][pid] for pid in group})!=1: raise ValueError('Cross-split duplicate')

    old = C.checked(previous)
    required = [previous/'manifest.json', previous/'region-short-teacher-fix-v1.json']
    for seed in RULE['seeds']:
        folder, _, _ = old_training(previous, seed)
        required += [folder/n for n in ('verified.json','proof.json','steps.json')]
    C.freeze(root/'plan.json', dict(rule=RULE, sources=sources(),
        previous_files={p.relative_to(previous).as_posix():C.sha(p) for p in required},
        ids=old['ids'], images=old['images'], external=ext, external_manifest_sha256=C.sha(audit/'data-manifest.json')))
    print('PLAN VERIFIED: 8 pilot arms; paired inputs; TrackRAD cached teachers / LLD synthetic geometry; no native LLD video', flush=True)


def checked(root, previous):
    plan = C.read(root/'plan.json')
    if plan['rule'] != RULE or plan['sources'] != sources():
        raise ValueError('Study source/rule changed; use a new output root')
    C.verify(previous, plan['previous_files']); C.checked(previous)
    return plan


def receipt(root, path, files, **extra):
    C.save(path, dict(complete=True, plan=C.sha(root/'plan.json'),
        files={p.relative_to(root).as_posix():C.sha(p) for p in files}, **extra))


def cached(root, path):
    if not path.exists(): return None
    r = C.read(path)
    if not r.get('complete') or r['plan'] != C.sha(root/'plan.json'):
        raise ValueError('Stale domain-study receipt')
    C.verify(root, r['files'])
    return r


def decision(root, phase):
    folder = root/'reports'/phase/'validation-10'
    if not cached(root, folder/'verified.json'): raise ValueError('Validation report required')
    d = C.read(folder/'selection.json')
    if d['selection_split'] != 'validation-10': raise ValueError('Wrong selection split')
    return d


def arms_for(root, phase):
    if phase in ('smoke','pilot'): return list(ARMS)
    if phase != 'full': raise ValueError('Invalid phase')
    d = decision(root, 'pilot')
    return ['M0','F0'] + d['promoted'] if (d['promoted'] or d.get('partial_no_adversarial_qualifies_vs_M0',False)) else []


def seeds_for(phase):
    if phase not in ('smoke','pilot','full'): raise ValueError('Invalid phase')
    return RULE['seeds'] if phase == 'full' else [0]


def gate(root, phase, arm, seed):
    if arm not in arms_for(root, phase) or seed not in seeds_for(phase):
        raise ValueError('Arm/seed not authorized by this phase')
    if phase != 'smoke' and not cached(root, root/'smoke/audit.json'):
        raise ValueError('GPU smoke including deployment inference required')


def test_gate(root):
    d = decision(root, 'full')
    if d['test_arms'] != arms_for(root, 'full') or not d['test_arms']:
        raise ValueError('Invalid frozen test roster')
    return d['test_arms']
