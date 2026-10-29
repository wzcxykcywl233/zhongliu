"""Frozen protocol and receipts for mixed sparse-mask / point-region experiments."""
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys

PACKAGE = Path(__file__).resolve().parent
CHECKOUT = PACKAGE.parents[1]
APP = CHECKOUT / 'cotracker-algorithm'
sys.path[:0] = [str(PACKAGE), str(APP / 'ext/co-tracker'), str(APP),
               str(PACKAGE.parent / 'medcotracker-acdc-smoke-v1'),
               str(PACKAGE.parent / 'medcotracker-acdc-ab-v1')]
PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
ARMS = ('M0', 'W1', 'D1', 'G1')
RULE = dict(version=1, arms=list(ARMS), exclusions=['patient076', 'patient082'],
    seed_base=20261029, smoke_steps=6, pilot_steps=250, full_steps=2000,
    mix='alternating TrackRAD/ACDC; 1:1 optimizer updates', queries=128,
    trackrad_frames=60, acdc_frames='native ED-to-ES or ES-to-ED, inclusive; no padding',
    resize=[384, 512], lr=5e-5, weight_decay=1e-5, iterations=4,
    optimizer='AdamW, constant LR, clip10', weak_weight=.05, weak_distance_scale_pixels=8., region_weight=.1,
    initialization='official scaled + fresh compact Mamba; no B1 checkpoint reuse',
    trainable='Mamba only; D1/G1 additionally query-conditioned region head',
    train_intensity='both domains clip p1/p99 to 0..255; inference retains fixed historical pipeline',
    orientation='TrackRAD W,H,T -> T,H,W; ACDC T,Z,Y,X slice -> T,Y,X',
    query_source='query-frame mask only, longest contour, 128 points; no target-frame based selection',
    region_feature_size=[48,64], region_dominant='G1 removes teacher coordinate loss; keeps teacher visibility gating',
    weak='boundary distance, only annotated non-query frames and teacher-visible points',
    region='BCE + soft Dice on annotated non-query frames; first mask is prompt only',
    selection='DSC>0, D98>=-.002, HD95<=.02; full >=2/3 DSC-positive seeds; worst>=-.002',
    region_gate='pilot D1-region must qualify against M0-points before G1 optimization (except bounded smoke)',
    test='explicit test only, fixed full arms and outputs, no test selection',
    note='New mixed-training protocol, not an exact historical B1 reproduction. Public test already inspected.')
FIELDS = {'DSC':'dice_similarity_coefficient', 'HD95':'hausdorff_distance_95',
          'MASD':'surface_distance_average', 'CD':'center_distance', 'D98':'relative_d98_dose'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def freeze(path, value):
    if Path(path).exists():
        if read(path) != value: raise ValueError('Frozen artifact changed; NEW output root required: '+str(path))
    else: save(path, value)

def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe path')
    p = (Path(root)/relative).resolve()
    if Path(root).resolve() not in p.parents: raise ValueError('Escaping path')
    return p

def verify(root, files):
    for p, h in files.items():
        if sha(safe(root,p)) != h: raise ValueError('Changed artifact: '+p)

def inventory(prior, split):
    result = {}
    for r in prior['Dataset']:
        if r['Split'] != split: continue
        parts = PureWindowsPath(r['File']).parts
        if parts.count(r['Case']) != 1: raise ValueError('Ambiguous dataset path')
        name = '/'.join(parts[parts.index(r['Case']):])
        if name in result: raise ValueError('Duplicate inventory entry')
        result[name] = r['SHA256'].lower()
    return result

def sources():
    folders = [PACKAGE, APP, PACKAGE.parent/'teacher-combination-search-v4',
               PACKAGE.parent/'teacher-input-search-v2', PACKAGE.parent/'teacher-pool-search-v1',
               PACKAGE.parent/'teacher-ensemble-search-v3', PACKAGE.parent/'teacher-single-factor-v1',
               PACKAGE.parent/'medcotracker-acdc-ab-v1', PACKAGE.parent/'medcotracker-acdc-smoke-v1']
    return {p.relative_to(CHECKOUT).as_posix():sha(p) for d in folders for p in sorted(d.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.ps1','.txt','.json') and '__pycache__' not in p.parts}

def prepare(root, prior, audit, acdc, train, validation, weights, images):
    import smoke as S
    original = read(prior/'frozen-study.json'); ext = read(audit/'data-manifest.json')
    S.make_jobs(ext)  # original native-v2 100-patient audit must be intact
    ids = {s:sorted({r['Case'] for r in original['Dataset'] if r['Split']==s})
           for s in ('train-40','validation-10','test-38')}
    if [len(ids[s]) for s in ids] != [40,10,38] or len(set(sum(ids.values(),[]))) != 88:
        raise ValueError('Require disjoint 40/10/38')
    files = {s:inventory(original,s) for s in ids}
    verify(train,files['train-40']); verify(validation,files['validation-10'])
    rows = ext['patients']; lookup = {r['patient']:r for r in rows}
    if lookup['patient074']['split'] != 'external-dev-20' or any(
        lookup[p]['split'] != 'external-train-80' for p in RULE['exclusions']):
        raise ValueError('QC patient/split binding changed')
    external_files = {'training/'+r['patient']+'/'+n:h for r in rows for n,h in r['files_sha256'].items()}
    verify(acdc, external_files)
    eligible = [r['patient'] for r in rows if r['split']=='external-train-80' and r['patient'] not in RULE['exclusions']]
    if len(eligible)!=78: raise ValueError('Expected 78 ACDC training patients')
    pool = S.load_protocol(CHECKOUT).baseline()
    hashes = {r['checkpoint']:sha(weights/r['checkpoint']) for r in pool}
    # Bind to successful historical smoke weight receipts, without importing stale source fingerprints.
    prior_weights = {r['Name']:r['SHA256'].lower() for r in original.get('Weights',[])}
    if not prior_weights:
        raise ValueError('Historical study lacks weight provenance')
    if any(prior_weights.get(n)!=h for n,h in hashes.items()): raise ValueError('Unverified weights')
    value = dict(rule=RULE, source=sources(), weights=hashes, pool=pool, images=images,
        prior_sha256=sha(prior/'frozen-study.json'), audit_sha256=sha(audit/'data-manifest.json'),
        external_files=external_files, eligible=sorted(eligible), external_dev=sorted(r['patient'] for r in rows if r['split']=='external-dev-20'),
        ids=ids, files=files, exclusions_reason={'patient082':'observed cine jump; whole-case exclusion',
            'patient076':'decoded cine duplicates dev patient074; keep dev split unchanged'})
    freeze(root/'manifest.json',value)
    print('PLAN VERIFIED 78 ACDC train / 20 dev; TrackRAD 40/10/38; no test access',flush=True)
    return value

def checked(root):
    p = read(root/'manifest.json')
    if p['rule']!=RULE or p['source']!=sources(): raise ValueError('Protocol/source changed')
    return p

def receipt(root, path, files, **extra):
    save(path,dict(complete=True,manifest_sha256=sha(root/'manifest.json'),
                  files={p.relative_to(root).as_posix():sha(p) for p in files},**extra))

def cached(root,path):
    if not path.exists(): return None
    r=read(path)
    if not r.get('complete') or r['manifest_sha256']!=sha(root/'manifest.json'):
        raise ValueError('Stale receipt')
    verify(root,r['files']); return r

def arm_outputs(arm):
    return ['points','region'] if arm=='D1' else ['region'] if arm=='G1' else ['points']

def qualifies(rows):
    import statistics
    import math
    if len(rows) not in (1,3) or not all(math.isfinite(float(r[m])) for r in rows for m in FIELDS):
        raise ValueError('Selection needs finite one/three-seed paired deltas')
    mean={m:statistics.mean(r[m] for r in rows) for m in FIELDS}
    return (mean['DSC']>0 and mean['D98']>=-.002 and mean['HD95']<=.02 and
            sum(r['DSC']>0 for r in rows)>=(2 if len(rows)==3 else 1) and min(r['DSC'] for r in rows)>=-.002)
