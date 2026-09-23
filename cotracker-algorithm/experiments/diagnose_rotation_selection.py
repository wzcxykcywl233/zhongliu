"""Read-only real-log audit plus known-angle synthetic encoder diagnostics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from analyze_frame_backcheck import read_json, export_csv, sha256
# Use the committed diagnostic helper mounted beside this script, not a package
# installer; the inference model and weights come from the recorded image.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'resources'))
from rotation_backcheck import ANGLES, crop, rotate, decide, sample, encode_centers


def accepted_gain(record):
    n=int(record['supported_points'])
    a=int(record['angle_accepted_points'])
    if not 0<=a<=n or a+int(record['angle_fallback_points'])!=n:
        raise ValueError('Invalid accepted/fallback counts')
    if not n: return 0.,0,0
    difference=float(record['patch_aligned'])-float(record['patch_zero'])
    if not np.isfinite(difference): raise ValueError('Nonfinite score')
    if not a and abs(difference)>2e-6:
        raise ValueError('Zero-angle fallback changed score; cannot reconstruct accepted gain')
    # Means stored as float32: exact algebra, with rounding uncertainty. These
    # are point-frame weighted gains, NOT a per-point win rate/distribution.
    return difference*n if a else 0.,a,n


def audit_logs(root,out):
    cases=sorted((root/'rotation_history_i2'/'checkpoint'/'jobs').glob('*/diagnostics.json'))
    if len(cases)!=38: raise ValueError('Expected exactly 38 complete test cases')
    rows=[]
    for path in cases:
        d=read_json(path)
        if d['profile']!='rotation_history_i2': raise ValueError('Wrong profile')
        other=read_json(root/'rotation_control_i2'/'checkpoint'/'jobs'/path.parent.name/'diagnostics.json')
        if not d['prediction']['array_sha256'] or d['prediction']['array_sha256']!=other['prediction']['array_sha256']:
            raise ValueError('Control output mismatch')
        for profile,data in (('rotation_history_i2',d),('rotation_control_i2',other)):
            image=root/profile/'checkpoint'/'jobs'/path.parent.name/'output'/'images'/'mri-linac-series-targets'/'output.mha'
            if sha256(image)!=data['prediction']['output_file_sha256']:
                raise ValueError('Prediction hash mismatch')
        for r in d['frame_backcheck']:
            gain,a,n=accepted_gain(r)
            rows.append(dict(Case=path.parent.name,Frame=r['frame'],Supported=n,Accepted=a,
                BoundaryAccepted=r['angle_-20_points']+r['angle_20_points'],
                AcceptedGainSum=gain,
                AcceptedMeanGain=gain/a if a else None,
                GainRoundingBound=2e-6*n/a if a else None,
                OriginalMainEligible=n>=4))
    export_csv(out/'real-frame-accepted-gains.csv',rows)
    percase=[]
    for case in sorted({r['Case'] for r in rows}):
        for scope in ('all_supported','original_main_eligible'):
            group=[r for r in rows if r['Case']==case and (scope=='all_supported' or r['OriginalMainEligible'])]
            accepted=sum(r['Accepted'] for r in group)
            percase.append(dict(Case=case,Scope=scope,Accepted=accepted,
                AcceptedMeanGain=sum(r['AcceptedGainSum'] for r in group)/accepted if accepted else None))
    export_csv(out/'real-case-accepted-gains.csv',percase)
    summaries=[]
    for scope in ('all_supported','original_main_eligible'):
        group=[r for r in rows if scope=='all_supported' or r['OriginalMainEligible']]
        accepted=sum(r['Accepted'] for r in group)
        supported=sum(r['Supported'] for r in group)
        summaries.append(dict(Scope=scope,Supported=supported,Accepted=accepted,
            AcceptedFraction=accepted/supported if supported else None,
            BoundaryFraction=sum(r['BoundaryAccepted'] for r in group)/accepted if accepted else None,
            AcceptedMeanGain=sum(r['AcceptedGainSum'] for r in group)/accepted if accepted else None,
            GainRoundingBound=2e-6*supported/accepted if accepted else None))
    return summaries


@torch.no_grad()
def synthetic(model,out):
    device=next(model.parameters()).device
    rows=[]
    truth_angles=(-30.,-20.,-10.,-5.,0.,5.,10.,20.,30.)
    offsets=((2.,1.),(8.,4.),(12.,-6.))
    center=torch.tensor([[63.5,63.5]],device=device)
    q=torch.tensor([[[31.5,31.5]]],device=device)
    for seed in range(4):
        generator=torch.Generator(device=device).manual_seed(20260923+seed)
        for texture in ('smooth_random','stripes','flat'):
            checkpoint=out/f'synthetic-{seed}-{texture}.json'
            if checkpoint.exists():
                rows.extend(read_json(checkpoint))
                print(f'SKIP synthetic seed={seed} texture={texture}',flush=True)
                continue
            if texture=='smooth_random':
                image=F.avg_pool2d(torch.rand(1,1,128,128,device=device,generator=generator),5,1,2)[0]*255
            elif texture=='stripes':
                axis=torch.arange(128,device=device).float()
                yy,xx=torch.meshgrid(axis,axis,indexing='ij')
                image=(127.5+90*torch.sin(.29*xx+.11*yy+seed))[None]
            else:
                image=torch.full((1,128,128),127.5,device=device)
            def features(patch):
                return F.normalize(model.fnet((2*patch/255-1).expand(-1,3,-1,-1)).float(),dim=1,eps=1e-6)
            reference,_=crop(image,center,0)
            ref=features(reference)
            qvectors=[]
            for angle in ANGLES:
                patch,_=crop(image,center,angle)
                qvectors.append(encode_centers(model,patch)[0])
            qvectors=torch.stack(qvectors)
            batch=[]
            for truth in truth_angles:
                target,_=crop(image,center,truth)
                fmap=features(target)
                target_center=sample(fmap,q,model.stride)[0,0]
                for offset in offsets:
                    delta=torch.tensor(offset,device=device)
                    target_probe=sample(fmap,q+delta,model.stride)
                    scores=torch.stack([(sample(ref,q+rotate(delta,-a),model.stride)*target_probe).sum(-1).clamp(-1,1).squeeze()
                                        for a in ANGLES])
                    chosen,accept,gain,gap=decide(scores[None])
                    best=int(scores.argmax())
                    selected=int(chosen[0])
                    zero=float((qvectors[2]*target_center).sum().clamp(-1,1))
                    aligned=float((qvectors[selected]*target_center).sum().clamp(-1,1))
                    record=dict(Seed=seed,Texture=texture,TrueAngle=truth,OffsetX=offset[0],OffsetY=offset[1],
                        InSearchRange=abs(truth)<=20,OnSearchGrid=truth in ANGLES,
                        RawAngle=ANGLES[best],ChosenAngle=ANGLES[selected],Accepted=bool(accept[0]),
                        RawAbsoluteError=abs(ANGLES[best]-truth),ChosenAbsoluteError=abs(ANGLES[selected]-truth),
                        ZeroAbsoluteError=abs(truth),BoundaryRaw=abs(ANGLES[best])==20,
                        PeakGain=float(gain[0]),PeakGap=float(gap[0]),CenterGain=aligned-zero)
                    record.update({f'Score_{int(a)}':float(scores[i]) for i,a in enumerate(ANGLES)})
                    batch.append(record)
            temporary=checkpoint.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(batch,indent=2,allow_nan=False),encoding='utf-8')
            temporary.replace(checkpoint)
            rows.extend(batch)
            print(f'DONE synthetic seed={seed} texture={texture} trials={len(batch)}',flush=True)
    export_csv(out/'synthetic-angle-trials.csv',rows)
    groups=[]
    for texture in ('smooth_random','stripes','flat'):
        for scope in ('in_range_grid','in_range_offgrid','outside_range'):
            r=[x for x in rows if x['Texture']==texture and
                ((scope=='in_range_grid' and x['OnSearchGrid']) or
                 (scope=='in_range_offgrid' and x['InSearchRange'] and not x['OnSearchGrid']) or
                 (scope=='outside_range' and not x['InSearchRange']))]
            groups.append(dict(Texture=texture,Scope=scope,Trials=len(r),
                RawMAE=float(np.mean([x['RawAbsoluteError'] for x in r])),
                ChosenMAE=float(np.mean([x['ChosenAbsoluteError'] for x in r])),
                ZeroMAE=float(np.mean([x['ZeroAbsoluteError'] for x in r])),
                NonzeroAcceptedRate=float(np.mean([x['Accepted'] for x in r])),
                BoundaryRawRate=float(np.mean([x['BoundaryRaw'] for x in r]))))
    export_csv(out/'synthetic-angle-summary.csv',groups)
    return groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--image-id',required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=Path('/opt/app/torch/hub/checkpoints/scaled_offline.pth')
    source=Path(__file__).resolve().parents[1]
    inputs={'image_id':args.image_id,'checkpoint':sha256(checkpoint),
        'script':sha256(Path(__file__)),'helper':sha256(source/'resources'/'rotation_backcheck.py'),
        'sampling_helper':sha256(source/'resources'/'frame_backcheck.py'),
        'analysis_helper':sha256(source/'experiments'/'analyze_frame_backcheck.py'),
        'logs':{str(p.relative_to(args.results)):sha256(p) for p in sorted(args.results.glob('*/checkpoint/jobs/*/diagnostics.json'))}}
    frozen=args.output/'frozen-diagnostic.json'
    if frozen.exists() and read_json(frozen)!=inputs: raise ValueError('Diagnostic inputs changed; use new output directory')
    if not frozen.exists():
        temporary=frozen.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(inputs,indent=2),encoding='utf-8')
        temporary.replace(frozen)
    real=audit_logs(args.results,args.output)
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'model' in state: state=state['model']
    if 'model_state_dict' in state: state=state['model_state_dict']
    model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    torch.set_num_threads(2)
    synthetic(model,args.output)
    summary={'real_test38':real,'complete':True,
        'limitations':'Real gain reconstructed from float32 frame means, not per-point win rates. Conditional accepted subset is selected, not causal. Synthetic textures and known image warps are NOT MRI angle ground truth; correlated trials, no p-values. Flat patterns have no identifiable physical angle.'}
    temp=args.output/'selection-diagnostic-summary.json.tmp'
    temp.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(args.output/'selection-diagnostic-summary.json')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
