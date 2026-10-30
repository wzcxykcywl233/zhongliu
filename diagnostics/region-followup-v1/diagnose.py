"""Fixed D1 weights/tracks. Diagnostic Dice is not official DSC or tracking GT."""
from common import *
import numpy as np
import torch
import torch.nn.functional as F
import reports as R
import evaluate as E
from region import RegionHead, prior_and_heat
from train import features


def normalize(video):
    lo, hi = np.percentile(video, [1,99]).astype(np.float32)
    if not np.isfinite([lo,hi]).all() or hi <= lo: raise ValueError('Constant/nonfinite diagnostic cine')
    return np.clip((video-lo)/(hi-lo),0,1)*np.float32(255)


def frame_rows(output, gt, seed, case, mode):
    if output.shape != gt.shape or output.ndim != 3: raise ValueError('Diagnostic mask shape mismatch')
    output, gt = output>0, gt>0
    rows = []
    for t in range(1, gt.shape[2]):
        p, g = output[:,:,t], gt[:,:,t]
        pa, ga = int(p.sum()), int(g.sum())
        center = float(np.linalg.norm(np.argwhere(p).mean(0)-np.argwhere(g).mean(0))) if pa and ga else None
        rows.append(dict(Seed=seed, Case=case, Mode=mode, Frame=t,
            TemporalBin=min(3, int(4*(t-1)/max(1,gt.shape[2]-1))),
            DiagnosticDice=2*int((p&g).sum())/(pa+ga) if pa+ga else 1.,
            PredArea=pa, GtArea=ga, AreaRatio=pa/ga if ga else None,
            PredEmpty=pa==0, GtEmpty=ga==0, CentroidDistancePixels=center))
    return rows


def native_masks(chunks, array, query):
    import resources
    masks = torch.cat(chunks)[:,0][None]  # B,T,H,W, not B,T,1,H,W
    result = resources.reshape_video(masks, target_shape=(array.shape[1], array.shape[0]))[0]
    output = result.cpu().numpy().transpose(2,1,0)
    from experiments import get_experiment_config
    # Follow-up is tied to this exact historical profile (including query lock).
    import os
    old = os.environ.get('COTRACKER_EXPERIMENT')
    os.environ['COTRACKER_EXPERIMENT'] = C.PROFILE
    try: _, config = get_experiment_config()
    finally:
        if old is None: os.environ.pop('COTRACKER_EXPERIMENT', None)
        else: os.environ['COTRACKER_EXPERIMENT'] = old
    if config.lock_first_mask: output[:,:,0] = query[:,:,0]>0
    return output


def decode(model, head, array, query, tracks, mode):
    if mode == 'static': return np.repeat(query>0, array.shape[2], axis=2), []
    if mode not in ('raw','percentile','prior'): raise ValueError('Unknown decoder mode')
    video = array.transpose(2,1,0).copy()
    if mode == 'percentile': video = normalize(video)
    video = F.interpolate(torch.from_numpy(video).float()[:,None], size=C.RULE['resize'],
        mode='bilinear', align_corners=False).expand(-1,3,-1,-1)
    first = F.interpolate(torch.from_numpy(query[:,:,0].T.copy()).float()[None,None],
        size=C.RULE['resize'], mode='nearest')[0].cuda()
    tracks = tracks.cuda(); chunks = []; diagnostics = []
    with torch.no_grad():
        qfeat = features(model, video[:1].cuda()) if mode != 'prior' else None
        for i in range(0,len(video),4):
            tr = tracks[i:i+4]
            prior, _ = prior_and_heat(first, tr, C.RULE['region_feature_size'], tracks[0])
            prior_logit = torch.logit(prior.clamp(.05,.95))
            if mode == 'prior': logits = prior_logit
            else:
                feat = features(model, video[i:i+4].cuda())
                logits = head(feat, first, tr, qfeat, tracks[0])
                for j in range(len(tr)):
                    diagnostics.append(dict(Frame=i+j, Mode=mode,
                        FeatureRms=float(feat[j].square().mean().sqrt()),
                        ResidualRms=float((logits[j]-prior_logit[j]).square().mean().sqrt()),
                        CoarseSignFlipFraction=float(((logits[j]>0)!=(prior_logit[j]>0)).float().mean())))
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite diagnostic logits')
            chunks.append((F.interpolate(logits, size=C.RULE['resize'], mode='bilinear', align_corners=False)>0).cpu())
    return native_masks(chunks, array, query), diagnostics


def run(root, previous, data, weights, seed):
    import SimpleITK as sitk
    import resources
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    plan = checked(root, previous)
    C.verify(data, C.read(previous/'manifest.json')['files']['validation-10'])
    cp = E.checkpoint(root, previous, weights, 'D1', seed)
    model = setup_model(str(cp), device='cuda').requires_grad_(False)
    configure_sidecar_runtime(model, backend='compact')
    head = RegionHead().cuda().eval()
    head.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['region_head'])
    inf = E.inference_module()
    for case in plan['ids']['validation-10']:
        folder = root/f'diagnosis/seed_{seed}'/case
        if cached(root, folder/'verified.json'):
            print(f'CACHE VERIFIED diagnosis seed={seed}/{case}',flush=True); continue
        image, array, query, meta = E.input_case(data, case, inf)
        traces = []; converter = resources.convert_point_trajectory_to_mask_sequence
        def capture(trajectories, **kw):
            traces.append(trajectories.detach().cpu().clone())
            return converter(trajectories, **kw)
        resources.convert_point_trajectory_to_mask_sequence = capture
        try:
            inf.seed_all(C.RULE['seed_base'])
            point = inf.run_algorithm(model, None, array, query, meta, 'points')
        finally: resources.convert_point_trajectory_to_mask_sequence = converter
        if len(traces) != 1 or traces[0].shape[:2] != (1,array.shape[2]):
            raise ValueError('Unexpected trajectory capture')
        folder.mkdir(parents=True,exist_ok=True)
        torch.save(traces[0], folder/'tracks.pth')
        E.write_mask(folder/'points.mha', point, image)
        outputs = {}; features_rows = []
        for mode in RULE['diagnostic_modes']:
            output, frows = decode(model, head, array, query, traces[0][0], mode)
            E.write_mask(folder/(mode+'.mha'), output, image)
            outputs[mode] = output
            features_rows += [dict(Seed=seed, Case=case, **r) for r in frows]
        # Future target masks are read only after every candidate output has been fixed.
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(data/case/'targets'/f'{case}_labels.mha')))
        rows = frame_rows(point, gt, seed, case, 'points')
        for mode, output in outputs.items(): rows += frame_rows(output,gt,seed,case,mode)
        R.csv_save(folder/'frame-metrics.csv',rows)
        R.csv_save(folder/'feature-metrics.csv',features_rows)
        replay = {}
        for view, output in [('points', point), ('region', outputs['raw'])]:
            old = previous/f'full/validation-10/seed_{seed}/D1'/view/'jobs'/case
            if not C.cached(previous, old/'verified.json'): raise ValueError('Old prediction receipt missing')
            original = sitk.GetArrayFromImage(sitk.ReadImage(str(old/'output/images/mri-linac-series-targets/output.mha')))
            if original.shape != output.shape: raise ValueError('Old/new mask shape mismatch')
            replay[view] = dict(changed_voxels=int(np.count_nonzero((original>0)!=(output>0))),
                                old_mask_sha256=C.sha(old/'output/images/mri-linac-series-targets/output.mha'))
        C.save(folder/'replay.json', dict(checkpoint=C.sha(cp), comparisons=replay,
            source_intensity_min=float(array.min()), source_intensity_max=float(array.max()),
            note='Nonzero replay differences mean this is not an exact old-output reproduction; variants still share the captured tracks.'))
        receipt(root,folder/'verified.json',[folder/n for n in ['tracks.pth','points.mha','frame-metrics.csv',
            'feature-metrics.csv','replay.json']+[m+'.mha' for m in RULE['diagnostic_modes']]],seed=seed,case=case)
        print(f'DIAGNOSE seed={seed}/{case}; replay changed voxels={replay}',flush=True)


def report(root, previous):
    import csv
    import statistics as st
    plan = checked(root, previous); allrows = []; files = []; replay = []
    training = []
    for seed in RULE['seeds']:
        _, steps, _ = training_receipt(previous,'D1',seed)
        for start in range(0, len(steps), 250):
            for domain in ('trackrad','acdc'):
                chunk = [s for s in steps[start:start+250] if s['domain']==domain]
                training.append(dict(Seed=seed,StartStep=start,Domain=domain,N=len(chunk),
                    **{k:st.mean(r[k] for r in chunk) for k in ('loss','point','weak','region','teacher_visible_fraction',
                        'query_spread_pixels','mean_predicted_spread_pixels','empty_annotated_target_frames')}))
        for case in plan['ids']['validation-10']:
            folder = root/f'diagnosis/seed_{seed}'/case
            if not cached(root,folder/'verified.json'): raise ValueError('Incomplete D1 diagnosis')
            files.append(folder/'verified.json')
            with (folder/'frame-metrics.csv').open(encoding='utf-8-sig',newline='') as f: allrows+=list(csv.DictReader(f))
            replay.append(dict(seed=seed,case=case,**C.read(folder/'replay.json')))
    cases = []
    for seed in RULE['seeds']:
        for case in plan['ids']['validation-10']:
            for mode in ['points']+RULE['diagnostic_modes']:
                r = [x for x in allrows if int(x['Seed'])==seed and x['Case']==case and x['Mode']==mode]
                cases.append(dict(Seed=seed,Case=case,Mode=mode,Frames=len(r),
                    DiagnosticDice=st.mean(float(x['DiagnosticDice']) for x in r),
                    PredEmptyFraction=st.mean(x['PredEmpty']=='True' for x in r)))
    summaries = []
    for mode in ['points']+RULE['diagnostic_modes']:
        r = [x for x in cases if x['Mode']==mode]
        summaries.append(dict(Mode=mode,SeedCasePairs=len(r),MacroCaseDiagnosticDice=st.mean(x['DiagnosticDice'] for x in r),
                             MacroEmptyFraction=st.mean(x['PredEmptyFraction'] for x in r)))
    temporal = []
    for mode in ['points']+RULE['diagnostic_modes']:
        for b in range(4):
            r = [x for x in allrows if x['Mode']==mode and int(x['TemporalBin'])==b]
            temporal.append(dict(Mode=mode,TemporalBin=b,Frames=len(r),
                MicroFrameDiagnosticDice=st.mean(float(x['DiagnosticDice']) for x in r)))
    folder = root/'reports/diagnosis/validation-10'
    for name, values in [('summary',summaries),('cases',cases),('frames',allrows),('temporal',temporal),('training-phases',training)]:
        path=folder/(name+'.csv'); R.csv_save(path,values); files.append(path)
    C.save(folder/'replays.json',replay); files.append(folder/'replays.json')
    receipt(root,folder/'verified.json',files,split='validation-10',diagnostic_only=True,
        note='Query excluded; empty/empty Dice=1. Diagnostic Dice/centroids use native pixels, not official DSC or tracking accuracy. '
             'Percentiles use whole cine, whereas training used clips. No causal conclusion or automatic model replacement.')
    print('REPORT COMPLETE D1 diagnosis; no training or test data',flush=True)
