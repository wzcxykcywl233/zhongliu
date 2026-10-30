"""Unchanged point deployment / official evaluator; validation-only selection."""
import base64
import statistics
from datetime import datetime, timezone
import numpy as np
import torch
from protocol import *
from train_domain import audit_training
import reports as R


def inference_module():
    return module('domain_shape',PACKAGE.parent/'region-output-shape-fix-v1/infer_fixed.py').load_fixed()


def allowed(root,phase,split,arm,seed):
    if split not in ('validation-10','test-38'): raise ValueError('Inference split forbidden')
    arms=test_gate(root) if split=='test-38' else arms_for(root,phase)
    if split=='test-38' and phase!='full': raise ValueError('Only full checkpoints test')
    if arm=='pretrained':
        if seed!=0 or phase=='smoke': raise ValueError('Reference once outside smoke')
    elif arm not in arms or seed not in seeds_for(phase): raise ValueError('Unauthorized output')


def infer(root,previous,data,weights,phase,split,arm,seed):
    plan=checked(root,previous); allowed(root,phase,split,arm,seed)
    audit_training(root,previous,phase)
    old=C.read(previous/'manifest.json'); C.verify(data,old['files'][split]); C.verify(weights,old['weights'])
    cp=weights/'scaled_offline.pth' if arm=='pretrained' else root/phase/f'train/seed_{seed}'/arm/'final.pth'
    folder=root/phase/split/f'seed_{seed}'/arm
    ids=plan['ids'][split][:2] if phase=='smoke' else plan['ids'][split]
    C.freeze(folder/'inputs.json',dict(plan=C.sha(root/'plan.json'),checkpoint=C.sha(cp),
        phase=phase,split=split,arm=arm,seed=seed,ids=ids,profile=C.PROFILE,output='points'))
    if cached(root,folder/'inference-verified.json'):
        print(f'CACHE VERIFIED {phase}/{split}/{seed}/{arm}',flush=True); return
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    import SimpleITK as sitk
    inf=inference_module(); model=setup_model(str(cp),device='cuda').requires_grad_(False)
    if arm!='pretrained': configure_sidecar_runtime(model,backend='compact')
    predictions=[]; files=[folder/'inputs.json']
    for case in ids:
        job=folder/'jobs'/case; marker=job/'verified.json'
        output=job/'output/images/mri-linac-series-targets/output.mha'
        if not cached(root,marker):
            p=data/case
            image=sitk.ReadImage(str(p/'images'/f'{case}_frames.mha'))
            array=sitk.GetArrayFromImage(image).astype(np.float32)
            query=sitk.GetArrayFromImage(sitk.ReadImage(str(p/'targets'/f'{case}_first_label.mha')))
            if query.shape!=array.shape[:2]+(1,) or not np.isfinite(array).all(): raise ValueError('Invalid inputs')
            meta=dict(frame_rate=inf.scalar(p/'frame-rate.json','number'),
                magnetic_field_strength=inf.scalar(p/'b-field-strength.json','number'),
                scanned_region=inf.scalar(p/'scanned-region.json','string'))
            inf.seed_all(C.RULE['seed_base']); start=datetime.now(timezone.utc).isoformat()
            mask=inf.run_algorithm(model,None,array,query,meta,'points')
            torch.cuda.synchronize(); end=datetime.now(timezone.utc).isoformat()
            if mask.shape!=array.shape or not np.isfinite(mask).all() or not np.isin(mask,[0,1]).all():
                raise ValueError('Invalid output mask')
            output.parent.mkdir(parents=True,exist_ok=True)
            result=sitk.GetImageFromArray(mask.astype(np.uint8)); result.CopyInformation(image)
            sitk.WriteImage(result,str(output),True)
            if not np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(output))),mask): raise ValueError('Output roundtrip failed')
            C.save(job/'prediction.json',inf.prediction(case,meta,start,end))
            receipt(root,marker,[output,job/'prediction.json',folder/'inputs.json'],case=case)
        predictions.append(C.read(job/'prediction.json')); files += [output,job/'prediction.json',marker]
    C.save(folder/'predictions.json',predictions)
    receipt(root,folder/'inference-verified.json',files+[folder/'predictions.json'],
            ids=ids,phase=phase,split=split,arm=arm,seed=seed)
    print(f'INFERENCE COMPLETE {phase}/{split}/{seed}/{arm}',flush=True)


def record(root,previous,relative):
    checked(root,previous); folder=C.safe(root,relative)
    proof=cached(root,folder/'inference-verified.json')
    if not proof: raise ValueError('Missing inference proof')
    R.official(folder/'metrics.json',proof['ids'])
    raw=(folder/'evaluation.log').read_bytes()
    encoding='utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    payloads=[s.split('=',1)[1] for s in raw.decode(encoding).splitlines() if s.startswith('REGION_METRICS_BASE64=')]
    if not payloads or base64.b64decode(payloads[-1],validate=True)!=(folder/'metrics.json').read_bytes():
        raise ValueError('Official metrics byte transport mismatch')
    receipt(root,folder/'metrics-verified.json',[folder/'metrics.json',folder/'inference-verified.json'],byte_identical=True)


def selection(metrics,seeds,arms):
    def qualifies(a,b):
        return C.qualifies([{m:metrics[a,s][m]-metrics[b,s][m] for m in C.FIELDS} for s in seeds])
    winners=[]; eligible=[]
    for family,candidates in FAMILIES.items():
        reference='M0' if family=='sidecar' else 'F0'
        valid=[a for a in candidates if a in arms and qualifies(a,reference) and qualifies(a,'M0')]
        eligible += valid
        if valid:
            winners.append(sorted(valid,key=lambda a:(-statistics.mean(metrics[a,s]['DSC'] for s in seeds),
                statistics.mean(metrics[a,s]['HD95'] for s in seeds),a))[0])
    return dict(selection_split='validation-10',eligible=eligible,promoted=winners,
        winner_qualifies_vs_pretrained={a:qualifies(a,'pretrained') for a in winners},
        partial_no_adversarial_qualifies_vs_M0=qualifies('F0','M0'),rule=RULE['selection'])


def diagnostics(root,phase,arms,seeds,folder):
    rows=[]
    for seed in seeds:
        for arm in arms:
            data=C.read(root/phase/f'train/seed_{seed}'/arm/'steps.json')
            # Last 100 steps avoid warm-up. Accuracy is online training accuracy, not
            # a held-out domain probe and NOT a selection metric.
            tail=data[-100:]
            acc=[]
            for domain in ('trackrad','acdc'):
                subset=[r for r in tail if r['domain']==domain]
                domain_acc=[float(r['domain_correct']) for r in subset if 'domain_correct' in r]
                features=np.asarray([r['pooled_feature'] for r in subset if 'pooled_feature' in r])
                if domain_acc: acc.append(statistics.mean(domain_acc))
                rows.append(dict(Seed=seed,Arm=arm,Domain=domain,Steps=len(subset),
                    TaskLossMean=statistics.mean(r['point'] for r in subset),
                    DomainLossMean=statistics.mean(r['domain_loss'] for r in subset),
                    OnlineDomainAccuracy=statistics.mean(domain_acc) if domain_acc else '',
                    FeatureRmsMean=statistics.mean(r['feature_rms'] for r in subset if 'feature_rms' in r) if domain_acc else '',
                    WithinDomainClipFeatureSpread=float(features.var(axis=0).mean()**.5) if len(features)>1 else '',
                    StepSecondsMedian=statistics.median(r['seconds'] for r in data),
                    PeakMemoryGB=max(r['max_memory_bytes'] for r in data)/2**30))
            for r in rows[-2:]: r['OnlineBalancedDomainAccuracy']=statistics.mean(acc) if len(acc)==2 else ''
    path=folder/'training-diagnostics.csv'; R.csv_save(path,rows); return path


def report(root,previous,phase,split):
    plan=checked(root,previous)
    if phase=='smoke': raise ValueError('Smoke is not an accuracy report')
    arms=arms_for(root,phase)
    if split=='test-38': arms=test_gate(root)
    if not arms: raise ValueError('No eligible full experiment')
    audit_training(root,previous,phase)
    seeds=seeds_for(phase); metrics={}; cases={}; rows=[]; summary=[]; files=[]
    for arm in ['pretrained']+arms:
        group=[]
        for seed in ([0] if arm=='pretrained' else seeds):
            folder=root/phase/split/f'seed_{seed}'/arm
            inf=cached(root,folder/'inference-verified.json')
            if not inf or inf['ids']!=plan['ids'][split] or not cached(root,folder/'metrics-verified.json'):
                raise ValueError('Missing complete official endpoint')
            agg,case=R.official(folder/'metrics.json',plan['ids'][split])
            metrics[arm,seed]=agg; cases[arm,seed]=case
            row=dict(Split=split,Phase=phase,Arm=arm,Seed=seed,**agg); rows.append(row); group.append(row)
            files += [folder/'metrics.json',folder/'inference-verified.json',folder/'metrics-verified.json']
        result=dict(Split=split,Phase=phase,Arm=arm,N=len(group))
        for m in list(C.FIELDS)+['TimeSec']:
            result[m+'_Mean']=statistics.mean(r[m] for r in group)
            result[m+'_SD']=statistics.stdev(r[m] for r in group) if len(group)>1 else 0.
        summary.append(result)
        if arm=='pretrained':
            for seed in seeds: metrics[arm,seed]=metrics[arm,0]; cases[arm,seed]=cases[arm,0]
    pairs={(a,'M0') for a in arms if a!='M0'} | {(a,'pretrained') for a in arms}
    pairs |= {(a,'F0') for a in arms if a.startswith('F') and a!='F0'}
    deltas=[]; case_rows=[]
    for candidate,reference in sorted(pairs):
        for seed in seeds:
            prefix=dict(Split=split,Seed=seed,Candidate=candidate,Reference=reference)
            deltas.append(dict(**prefix,**{'Delta_'+m:metrics[candidate,seed][m]-metrics[reference,seed][m] for m in C.FIELDS}))
            for case in plan['ids'][split]:
                case_rows.append(dict(**prefix,Case=case,**{'Delta_'+m:cases[candidate,seed][case][m]-cases[reference,seed][case][m] for m in C.FIELDS}))
    folder=root/'reports'/phase/split
    for name,content in [('summary',summary),('per-seed',rows),('paired-deltas',deltas),('case-deltas',case_rows)]:
        path=folder/(name+'.csv'); R.csv_save(path,content); files.append(path)
    files.append(diagnostics(root,phase,arms,seeds,folder))
    if split=='validation-10':
        d=selection(metrics,seeds,arms)
        d.update(test_arms=arms if phase=='full' else [],no_test_selection=True,
            full_gate_open=bool(d['promoted'] or d['partial_no_adversarial_qualifies_vs_M0']),note=RULE['note'])
        C.freeze(folder/'selection.json',d); files.append(folder/'selection.json')
    receipt(root,folder/'verified.json',files,phase=phase,split=split)
    for row in summary:
        print(f'RESULT {row["Arm"]}: DSC={row["DSC_Mean"]:.6f}; D98={row["D98_Mean"]:.6f}',flush=True)
    print(f'REPORT COMPLETE {phase}/{split}',flush=True)
