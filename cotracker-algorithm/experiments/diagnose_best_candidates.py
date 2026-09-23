"""Read-only label-aware diagnostics, never inference or a new performance result."""
import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid
from functools import lru_cache

import numpy as np

PROFILES = ('pvc_control', 'pvc_vc_decay')
METRICS = {'DSC': 'dice_similarity_coefficient', 'HD95': 'hausdorff_distance_95',
           'MASD': 'surface_distance_average', 'CD': 'center_distance', 'D98': 'relative_d98_dose'}


def verify_helpers(runtime, reference):
    """Check current reference compatibility, not unavailable historical image identity."""
    hashes = {}
    for name in ('reshape.py', 'seg_to_tap.py', 'tap_to_seg.py'):
        def normalized(path):
            return hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
        actual = normalized(runtime / name)
        expected = normalized(reference / name)
        if actual != expected:
            raise ValueError('Diagnostic reconstruction helper mismatch: ' + name)
        hashes[name] = actual
    return hashes


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def export(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['no_records']
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return float(2 * np.count_nonzero(a & b) / total) if total else None


def shift_mask(mask, dy, dx):
    """Integer translation with zero fill; NEVER wrap around the image."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    sy, sx = max(0, -dy), max(0, -dx)
    ty, tx = max(0, dy), max(0, dx)
    hh, ww = h - abs(dy), w - abs(dx)
    out[ty:ty+hh, tx:tx+ww] = mask[sy:sy+hh, sx:sx+ww]
    return out


def geometry(pred, truth):
    result = {'FrameDice': dice(pred, truth), 'TruthArea': int(truth.sum()),
              'PredictionArea': int(pred.sum()), 'EmptyTruth': not bool(truth.any()),
              'EmptyPrediction': not bool(pred.any())}
    if not truth.any() or not pred.any():
        return {**result, 'CentroidDistancePixels': None, 'AreaRatio': None,
                'CentroidAlignedDice': None, 'CentroidGain': None,
                'ShiftY': None, 'ShiftX': None, 'ClippedPixels': None}
    ctrue, cpred = np.argwhere(truth).mean(0), np.argwhere(pred).mean(0)
    dy, dx = np.rint(ctrue - cpred).astype(int)
    aligned = shift_mask(pred, int(dy), int(dx))
    aligned_dice = dice(aligned, truth)
    return {**result, 'CentroidDistancePixels': float(np.linalg.norm(ctrue-cpred)),
            'AreaRatio': float(pred.sum()/truth.sum()), 'CentroidAlignedDice': aligned_dice,
            'CentroidGain': aligned_dice-result['FrameDice'], 'ShiftY': int(dy), 'ShiftX': int(dx),
            'ClippedPixels': int(pred.sum()-aligned.sum())}


def bootstrap(values, repeats=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing/nonfinite paired values')
    rng = np.random.default_rng(20260923)
    means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return {'Cases': len(values), 'MeanDelta': float(values.mean()),
            'MedianDelta': float(np.median(values)), 'BootstrapLow95': float(low),
            'BootstrapHigh95': float(high)}


def validate_configs(a, b):
    expected = {'hierarchical_span': 10, 'n_iterations': 2, 'support_grid_size': 0,
                'query_memory_mode': 'topk_confidence_diversity', 'query_memory_refinement': 'none'}
    for name, value in expected.items():
        if a.get(name) != value or b.get(name) != value:
            raise ValueError(f'Unexpected mainline config: {name}')
    if a.get('query_state_inheritance') != 'none' or b.get('query_state_inheritance') != 'vc_decay':
        raise ValueError('Wrong query-state inheritance modes')
    aa, bb = dict(a), dict(b)
    aa.pop('query_state_inheritance'); bb.pop('query_state_inheritance')
    if aa != bb:
        raise ValueError('Profiles differ beyond query-state inheritance')


def valid_job(path):
    try:
        record = read(path/'complete.json')
        return bool(record['files']) and all((path/k).is_file() and sha(path/k) == v
                                            for k, v in record['files'].items())
    except (OSError, ValueError, KeyError):
        return False


@lru_cache(maxsize=2)
def reconstruction_functions(root):
    funcs = {}
    for module in ('reshape', 'seg_to_tap', 'tap_to_seg'):
        spec = importlib.util.spec_from_file_location('diagnostic_'+module, root/f'{module}.py')
        obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
        funcs[module] = obj
    return funcs


def representation(mask, root):
    """Ground-truth contour roundtrip, NOT ground-truth point correspondences."""
    import torch
    funcs = reconstruction_functions(root)
    resize = funcs['reshape'].reshape_video
    points = funcs['seg_to_tap'].convert_mask_to_points
    fill = funcs['tap_to_seg'].convert_points_to_mask
    native = torch.from_numpy(mask.copy())[None, None]
    scaled = resize(native.float(), (384, 512))
    resize_only = resize(scaled > .5, mask.shape)[0, 0].numpy()
    result = {'ResizeOnlyDice': dice(resize_only, mask)}
    for count in (250, 500, 1000, 2000):
        q = points(scaled, count, add_t0=False)[0]
        rebuilt = fill(q, (384, 512))[None, None]
        back = resize(rebuilt, mask.shape)[0, 0].numpy()
        result[f'Contour{count}Dice'] = dice(back, mask)
    return result


def files_for(dataset, previous, case):
    case_dir = dataset/case
    truth = case_dir/'targets'/f'{case}_staple_labels.mha'
    if not truth.exists(): truth = case_dir/'targets'/f'{case}_labels.mha'
    files = {'truth': truth, 'image': case_dir/'images'/f'{case}_frames.mha',
             'region': case_dir/'scanned-region.json', 'field': case_dir/'b-field-strength.json'}
    for profile in PROFILES:
        job = previous/profile/'checkpoint'/'jobs'/case
        files[profile] = job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
        files[profile+'_diagnostic'] = job/'diagnostics.json'
    return files


def analyze_case(case, files, attempt, resource_root, sample_frames):
    import SimpleITK as sitk
    import cv2
    # Stored arrays are W,H,T; internal contour pipeline uses T,H,W.
    def load(path): return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).transpose(2, 1, 0)
    truth = load(files['truth']) > 0
    preds = {p: load(files[p]) > 0 for p in PROFILES}
    if len(truth) < 2 or any(v.shape != truth.shape for v in preds.values()):
        raise ValueError(f'Wrong paired mask dimensions: {case}')
    diags = {p: read(files[p+'_diagnostic']) for p in PROFILES}
    for profile in PROFILES:
        if diags[profile]['profile'] != profile:
            raise ValueError('Wrong diagnostic profile')
        if sha(files[profile]) != diags[profile]['prediction']['output_file_sha256']:
            raise ValueError(f'Prediction hash mismatch: {case}/{profile}')
    validate_configs(*(diags[p]['config'] for p in PROFILES))
    region, field = read(files['region']), read(files['field'])
    rows = []
    for t in range(1, len(truth)):
        for profile in PROFILES:
            rows.append({'Case': case, 'Profile': profile, 'Frame': t, 'Region': region,
                         'FieldStrength': field, 'VideoFrames': len(truth),
                         'TimeQuartile': min(4, 1+int(4*(t-1)/(len(truth)-1))),
                         'NominalSpanOffset': t % 10, **geometry(preds[profile][t], truth[t])})
    export(attempt/'frames.csv', rows)
    chosen = np.unique(np.linspace(1, len(truth)-1, min(sample_frames, len(truth)-1), dtype=int))
    representation_rows = []
    for t in chosen:
        entry = {'Case': case, 'Frame': int(t), 'Valid': False, 'Reason': ''}
        if not truth[t].any():
            entry['Reason'] = 'empty ground truth'
        else:
            try:
                entry.update(representation(truth[t], resource_root)); entry['Valid'] = True
            except (ValueError, TypeError) as exc:
                entry['Reason'] = 'contour construction failed: '+str(exc)
        representation_rows.append(entry)
    export(attempt/'representation.csv', representation_rows)
    # Fixed selection: two worst mainline frames plus two largest candidate drops.
    ref = {r['Frame']: r for r in rows if r['Profile'] == PROFILES[0] and r['FrameDice'] is not None}
    cand = {r['Frame']: r for r in rows if r['Profile'] == PROFILES[1] and r['FrameDice'] is not None}
    worst = sorted(ref, key=lambda t: ref[t]['FrameDice'])[:2]
    drops = sorted(ref.keys() & cand.keys(), key=lambda t: cand[t]['FrameDice']-ref[t]['FrameDice'])[:2]
    images = load(files['image'])
    if images.shape != truth.shape: raise ValueError('MRI/mask dimension mismatch')
    for t in sorted(set(worst+drops)):
        img = images[t].astype(float)
        lo, hi = np.percentile(img, [1, 99])
        gray = np.uint8(np.clip((img-lo)/max(hi-lo, 1e-8), 0, 1)*255)
        panels = []
        for profile in PROFILES:
            panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for mask, color in ((truth[t], (0,255,0)), (preds[profile][t], (0,0,255))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel, contours, -1, color, 1)
            panel = cv2.copyMakeBorder(panel, 45, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(panel, f'{profile} t={t}', (5,16), cv2.FONT_HERSHEY_SIMPLEX, .4,(255,255,255),1)
            cv2.putText(panel, 'GT green / prediction red', (5,34), cv2.FONT_HERSHEY_SIMPLEX, .4,(255,255,255),1)
            panels.append(panel)
        if not cv2.imwrite(str(attempt/f'frame-{t:05d}.png'), np.concatenate(panels, axis=1)):
            raise OSError('Cannot save overlay')
    atomic_json(attempt/'case.json', {'case': case, 'frames': len(truth)-1,
                'overlay_frames': sorted(set(worst+drops)), 'region': region, 'field': field})


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def summarize(output, cases, official, split):
    paired = []
    for case in cases:
        row = {'Split': split, 'Case': case}
        for short, key in METRICS.items():
            a, b = [float(official[p][case][key]) for p in PROFILES]
            if not np.isfinite([a,b]).all(): raise ValueError('Nonfinite official metric')
            row.update({f'Control_{short}': a, f'Candidate_{short}': b, f'Delta_{short}': b-a})
        paired.append(row)
    export(output/'official-case-pairs.csv', paired)
    stats = []
    for metric in METRICS:
        values = np.array([r['Delta_'+metric] for r in paired])
        favorable = values if metric in ('DSC','D98') else -values
        stats.append({'Metric': metric, **bootstrap(values), 'ImprovedCases': int((favorable>1e-8).sum()),
                      'WorsenedCases': int((favorable < -1e-8).sum()), 'TieTolerance': 1e-8,
                      'LeaveOneCaseOutMin': float(min((values.sum()-v)/(len(values)-1) for v in values)),
                      'LeaveOneCaseOutMax': float(max((values.sum()-v)/(len(values)-1) for v in values))})
    export(output/'paired-summary.csv', stats)
    frames, reps = [], []
    for case in cases:
        frames.extend(csv_rows(output/'jobs'/case/'frames.csv'))
        reps.extend(csv_rows(output/'jobs'/case/'representation.csv'))
    export(output/'diagnostic-frames.csv', frames)
    export(output/'representation-frames.csv', reps)
    groups = {}
    for r in frames:
        for dimension in ('All','TimeQuartile','NominalSpanOffset','Region','FieldStrength'):
            key = (r['Profile'],dimension,'all' if dimension == 'All' else r[dimension],r['Case'])
            groups.setdefault(key, []).append(r)
    case_groups=[]
    for (profile, dimension, value, case), group in groups.items():
        row={'Profile':profile,'Dimension':dimension,'Group':value,'Case':case,'Frames':len(group)}
        for metric in ('FrameDice','CentroidDistancePixels','AreaRatio','CentroidGain'):
            vals=[float(r[metric]) for r in group if r[metric] != '']
            row[metric]=float(np.mean(vals)) if vals else None
        case_groups.append(row)
    export(output/'diagnostic-case-groups.csv',case_groups)
    pooled={}
    for r in case_groups: pooled.setdefault((r['Profile'],r['Dimension'],r['Group']),[]).append(r)
    macro=[]
    for (profile,dimension,value),group in pooled.items():
        row={'Profile':profile,'Dimension':dimension,'Group':value,'Cases':len(group)}
        for metric in ('FrameDice','CentroidDistancePixels','AreaRatio','CentroidGain'):
            vals=[r[metric] for r in group if r[metric] is not None]
            row[metric]=float(np.mean(vals)) if vals else None
        macro.append(row)
    export(output/'diagnostic-group-summary.csv',macro)
    rep_summary=[]
    for key in ('ResizeOnlyDice','Contour250Dice','Contour500Dice','Contour1000Dice','Contour2000Dice'):
        means=[]; valid_count=0
        for case in cases:
            vals=[float(r[key]) for r in reps if r['Case']==case and r.get('Valid')=='True' and r.get(key,'')!='']
            if vals: means.append(float(np.mean(vals))); valid_count+=len(vals)
        rep_summary.append({'Probe':key,'Cases':len(means),'ValidFrames':valid_count,'AttemptedFrames':len(reps),
                            'MacroCaseDice':float(np.mean(means)) if means else None})
    export(output/'representation-summary.csv',rep_summary)
    atomic_json(output/'summary.json', {'complete':True,'split':split,'cases':len(cases),'profiles':PROFILES,
        'diagnostic_frames_excluding_query':len(frames)//2,
        'representation_failed_frames':sum(r.get('Valid')!='True' for r in reps),
        'note':'Official metrics copied, never recomputed. Other Dice values are diagnostic only. '
        'Centroid alignment uses ground truth and is neither deployable nor an upper bound. '
        'Contour roundtrip is not true point tracking or a strict ceiling. '
        'Nominal offsets ignore merged spans. Case bootstrap is exploratory, not patient-independent evidence; no p-values.'})


def main():
    parser=argparse.ArgumentParser()
    for name in ('dataset','previous','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--split',choices=('validation-10','test-38'),required=True)
    parser.add_argument('--image-id',required=True)
    parser.add_argument('--source-image-id',required=True)
    parser.add_argument('--reference-resources',type=Path,required=True)
    parser.add_argument('--resource-root',type=Path,default=Path('/opt/app/resources'))
    parser.add_argument('--sample-frames',type=int,default=12)
    args=parser.parse_args()
    if args.sample_frames < 1: raise ValueError('sample-frames must be positive')
    # Keep numerical analysis bounded; no GPU, model load, training, or package install.
    import torch
    torch.set_num_threads(2)
    source_identity = args.previous/'frozen-images.json'
    if read(source_identity).get('trackrad-algorithm-cotracker-algorithm') != args.source_image_id:
        raise ValueError('Prediction source image identity mismatch')
    reference_helpers = verify_helpers(args.resource_root,args.reference_resources)
    versions = {'python':sys.version}
    for package in ('torch','numpy','scipy','scikit-image','SimpleITK','opencv-python','opencv-python-headless'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    cases=sorted(p.name for p in args.dataset.iterdir() if p.is_dir())
    if len(cases)!=(38 if args.split=='test-38' else 10): raise ValueError('Unexpected dataset count')
    args.output.mkdir(parents=True,exist_ok=True)
    all_files={case:files_for(args.dataset,args.previous,case) for case in cases}
    official={}; inputs={}
    for profile in PROFILES:
        path=args.previous/profile/'metrics.json'
        metric=read(path)
        official[profile]={r['case_id']:r for r in metric['results']}
        if sorted(official[profile])!=cases or len(metric['results'])!=len(cases):
            raise ValueError('Official metric case IDs mismatch')
        inputs[profile+'/metrics.json']=sha(path)
    for case, files in all_files.items():
        print('VERIFY INPUT '+case,flush=True)
        for key,path in files.items(): inputs[case+'/'+key]=sha(path)
    helpers={p.name:sha(p) for p in (args.resource_root/n for n in ('reshape.py','seg_to_tap.py','tap_to_seg.py'))}
    frozen={'schema':2,'split':args.split,'image':args.image_id,'inputs':inputs,'helpers':helpers,
            'prediction_source_image':args.source_image_id,'runtime_override':args.image_id!=args.source_image_id,
            'source_identity_sha256':sha(source_identity),'dependency_versions':versions,
            'reference_helpers_normalized_sha256':reference_helpers,
            'compatibility_note':'Helpers match current repository reference; unavailable historical image equivalence is not asserted.',
            'script':sha(Path(__file__)),'sample_frames':args.sample_frames,'seed':20260923}
    identity=args.output/'frozen-run.json'
    if identity.exists():
        if read(identity)!=frozen: raise ValueError('Inputs/code/image changed; choose a new output directory')
    else:
        if (args.output/'jobs').exists(): raise ValueError('Unfingerprinted jobs found')
        atomic_json(identity,frozen)
    (args.output/'jobs').mkdir(exist_ok=True)
    (args.output/'.attempts').mkdir(exist_ok=True)
    for case in cases:
        destination=args.output/'jobs'/case
        if valid_job(destination): print('SKIP verified '+case,flush=True); continue
        if destination.exists(): destination.rename(args.output/'.attempts'/('incomplete-'+case+'-'+uuid.uuid4().hex))
        attempt=args.output/'.attempts'/(case+'-'+uuid.uuid4().hex); attempt.mkdir()
        print('DIAGNOSE '+case,flush=True)
        analyze_case(case,all_files[case],attempt,args.resource_root,args.sample_frames)
        for key,path in all_files[case].items():
            if sha(path)!=inputs[case+'/'+key]: raise ValueError('Source modified while analyzing')
        atomic_json(attempt/'complete.json',{'files':{p.name:sha(p) for p in attempt.iterdir() if p.is_file()}})
        attempt.rename(destination)
        print('COMMITTED '+case,flush=True)
    summarize(args.output,cases,official,args.split)
    print('COMPLETE '+str(args.output/'summary.json'),flush=True)


if __name__=='__main__': main()
