"""Fixed checkpoint evaluation. No optimization and no test-driven selection."""
from common import *
from datetime import datetime, timezone
import base64
import statistics
import numpy as np
import torch
import reports as R


def inference_module():
    return module('followup_shape', PACKAGE.parent/'region-output-shape-fix-v1/infer_fixed.py').load_fixed()


def checkpoint(root, previous, weights, arm, seed):
    if seed not in RULE['seeds']: raise ValueError('Invalid seed')
    if arm == 'pretrained':
        if seed != 0: raise ValueError('Reference runs once')
        C.verify(weights, C.read(previous/'manifest.json')['weights'])
        return weights/'scaled_offline.pth'
    source = root/'g1' if arm == 'G1' else previous
    folder, _, _ = training_receipt(source, arm, seed)
    if arm == 'G1' and not cached(root, folder/'followup-verified.json'):
        raise ValueError('G1 paired audit missing')
    return folder/'final.pth'


def input_case(data, case, inf):
    import SimpleITK as sitk
    p = data/case
    image = sitk.ReadImage(str(p/'images'/f'{case}_frames.mha'))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    query = sitk.GetArrayFromImage(sitk.ReadImage(str(p/'targets'/f'{case}_first_label.mha')))
    if query.shape != array.shape[:2]+(1,) or not np.isfinite(array).all():
        raise ValueError('Input shape/value invalid')
    meta = dict(frame_rate=inf.scalar(p/'frame-rate.json', 'number'),
        magnetic_field_strength=inf.scalar(p/'b-field-strength.json', 'number'),
        scanned_region=inf.scalar(p/'scanned-region.json', 'string'))
    return image, array, query, meta


def write_mask(path, output, image):
    import SimpleITK as sitk
    expected = sitk.GetArrayFromImage(image).shape
    if output.shape != expected or not np.isfinite(output).all() or not np.isin(output, [0,1]).all():
        raise ValueError('Invalid output mask')
    path.parent.mkdir(parents=True, exist_ok=True)
    result = sitk.GetImageFromArray(output.astype(np.uint8)); result.CopyInformation(image)
    sitk.WriteImage(result, str(path), True)
    if not np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), output):
        raise ValueError('MHA roundtrip failed')


def endpoint(arm, seed):
    if arm == 'G1': return f'g1/validation-10/seed_{seed}/G1/region', 'validation-10', 'region'
    if arm not in RULE['point_test']: raise ValueError('Unknown fixed endpoint')
    return f'points/test-38/seed_{seed}/{arm}/points', 'test-38', 'points'


def infer(root, previous, data, weights, arm, seed):
    plan = checked(root, previous)
    relative, split, view = endpoint(arm, seed)
    C.verify(data, C.read(previous/'manifest.json')['files'][split])
    cp = checkpoint(root, previous, weights, arm, seed)
    folder = root/relative
    C.freeze(folder/'inputs.json', dict(plan=C.sha(root/'followup-plan.json'), checkpoint=C.sha(cp),
        ids=plan['ids'][split], arm=arm, seed=seed, split=split, view=view, profile=C.PROFILE))
    if cached(root, folder/'inference-verified.json'):
        print('CACHE VERIFIED '+relative, flush=True); return
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    inf = inference_module()
    model = setup_model(str(cp), device='cuda').requires_grad_(False)
    if arm != 'pretrained': configure_sidecar_runtime(model, backend='compact')
    head = None
    if view == 'region':
        from region import RegionHead
        head = RegionHead().cuda().eval()
        head.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True)['region_head'])
    files = [folder/'inputs.json']; predictions = []
    for case in plan['ids'][split]:
        job = folder/'jobs'/case
        output_path = job/'output/images/mri-linac-series-targets/output.mha'
        marker = job/'verified.json'
        if not cached(root, marker):
            image, array, query, meta = input_case(data, case, inf)
            inf.seed_all(C.RULE['seed_base']); start = datetime.now(timezone.utc).isoformat()
            output = inf.run_algorithm(model, head, array, query, meta, view)
            torch.cuda.synchronize(); end = datetime.now(timezone.utc).isoformat()
            write_mask(output_path, output, image)
            C.save(job/'prediction.json', inf.prediction(case, meta, start, end))
            receipt(root, marker, [output_path, job/'prediction.json', folder/'inputs.json'], case=case)
        predictions.append(C.read(job/'prediction.json'))
        files += [output_path, job/'prediction.json', marker]
    C.save(folder/'predictions.json', predictions)
    receipt(root, folder/'inference-verified.json', files+[folder/'predictions.json'],
            ids=plan['ids'][split], split=split, arm=arm, seed=seed, view=view)
    print('INFERENCE COMPLETE '+relative, flush=True)


def record(root, previous, relative):
    checked(root, previous)
    folder = C.safe(root, relative)
    inf = cached(root, folder/'inference-verified.json')
    if not inf: raise ValueError('Missing inference proof')
    R.official(folder/'metrics.json', inf['ids'])
    data = (folder/'evaluation.log').read_bytes()
    encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    payloads = [s.split('=',1)[1] for s in data.decode(encoding).splitlines() if s.startswith('REGION_METRICS_BASE64=')]
    if not payloads or base64.b64decode(payloads[-1], validate=True) != (folder/'metrics.json').read_bytes():
        raise ValueError('Official metric transport byte mismatch')
    receipt(root, folder/'metrics-verified.json', [folder/'metrics.json', folder/'inference-verified.json'], byte_identical=True)


def report(root, previous, branch):
    plan = checked(root, previous)
    split = 'test-38' if branch == 'points' else 'validation-10'
    outputs = [('pretrained','points'), ('M0','points'), ('W1','points'), ('D1','points')]
    if branch == 'g1': outputs = [('M0','points'), ('D1','points'), ('D1','region'), ('G1','region')]
    values = {}; casevalues = {}; rows = []; summaries = []; files = []
    for arm, view in outputs:
        group = []
        for seed in ([0] if arm == 'pretrained' else RULE['seeds']):
            if branch == 'g1' and arm != 'G1':
                folder = previous/f'full/validation-10/seed_{seed}'/arm/view
                if not C.cached(previous, folder/'metrics-verified.json'): raise ValueError('Missing old official metrics')
            else:
                folder = root/endpoint(arm, seed)[0]
                if not cached(root, folder/'metrics-verified.json'): raise ValueError('Missing new official metrics')
                if not cached(root, folder/'inference-verified.json'): raise ValueError('Missing new predictions')
                files += [folder/'metrics.json', folder/'metrics-verified.json']
            agg, cases = R.official(folder/'metrics.json', plan['ids'][split])
            name = arm+'/'+view
            values[name,seed] = agg; casevalues[name,seed] = cases
            row = dict(Split=split, Arm=arm, Output=view, Seed=seed, **agg)
            rows.append(row); group.append(row)
        summary = dict(Split=split, Arm=arm, Output=view, N=len(group))
        for key in list(C.FIELDS)+['TimeSec']:
            summary[key+'_Mean'] = statistics.mean(x[key] for x in group)
            summary[key+'_SD'] = statistics.stdev(x[key] for x in group) if len(group)>1 else 0.
        summaries.append(summary)
    if branch == 'points':
        for seed in (1,2):
            values['pretrained/points',seed] = values['pretrained/points',0]
            casevalues['pretrained/points',seed] = casevalues['pretrained/points',0]
        pairs = [('W1/points','M0/points'), ('D1/points','M0/points'), ('D1/points','W1/points')]
        pairs += [(a+'/points','pretrained/points') for a in ('M0','W1','D1')]
    else:
        pairs = [('G1/region', ref) for ref in ('M0/points','D1/points','D1/region')]
    deltas = []; cases = []
    for candidate, reference in pairs:
        for seed in RULE['seeds']:
            prefix = dict(Split=split, Seed=seed, Candidate=candidate, Reference=reference)
            deltas.append(dict(**prefix, **{'Delta_'+k:values[candidate,seed][k]-values[reference,seed][k] for k in C.FIELDS}))
            for case in plan['ids'][split]:
                cases.append(dict(**prefix, Case=case, **{'Delta_'+k:casevalues[candidate,seed][case][k]-casevalues[reference,seed][case][k] for k in C.FIELDS}))
    folder = root/'reports'/branch/split
    for name, content in [('summary',summaries), ('per-seed',rows), ('paired-deltas',deltas), ('case-deltas',cases)]:
        path = folder/(name+'.csv'); R.csv_save(path, content); files.append(path)
    receipt(root, folder/'verified.json', files, split=split, branch=branch, no_selection=True,
            note=RULE['note'], g1_test_started=False)
    for row in summaries:
        print(f"RESULT {row['Arm']}/{row['Output']}: DSC={row['DSC_Mean']:.6f}; D98={row['D98_Mean']:.6f}", flush=True)
    print(f'REPORT COMPLETE {branch}/{split}', flush=True)
