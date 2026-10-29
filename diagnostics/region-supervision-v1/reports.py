"""Paired official endpoints. No test-dependent arm or output selection."""
import argparse
import csv
import math
import statistics
import base64
from core import *

def csv_save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def expected_arms(root,phase):
    if phase=='smoke': return list(ARMS)
    if phase=='pilot': return list(ARMS[:3])
    r=cached(root,root/'reports/pilot/validation-10/verified.json')
    if not r: raise ValueError('Pilot report required')
    decision=read(root/'reports/pilot/validation-10/selection.json')
    return list(ARMS if decision['region_gate'] else ARMS[:3])

def training_audit(root,phase,require_smoke_inference=False):
    plan=checked(root); arms=expected_arms(root,phase); seeds=[0] if phase!='full' else [0,1,2]
    keys=('video','queries','sampling','domain','slot','draw','label_tracks','label_visibility','label_confidence')
    files=[]
    for seed in seeds:
        reference=None; initial=None; head_initial=None
        for arm in arms:
            folder=root/phase/f'train/seed_{seed}'/arm
            proof=cached(root,folder/'verified.json')
            if not proof or proof['steps']!=RULE[phase+'_steps'] or not proof.get('reload_gpu_forward'):
                raise ValueError('Missing training/GPU handoff proof: '+str(folder))
            if (proof['phase'],proof['arm'],proof['seed'])!=(phase,arm,seed): raise ValueError('Wrong training receipt identity')
            rows=read(folder/'steps.json'); info=read(folder/'proof.json')
            if not info['frozen_backbone_exact'] or len(rows)!=proof['steps']: raise ValueError('Incomplete training')
            if [r['step'] for r in rows]!=list(range(proof['steps'])): raise ValueError('Duplicated/missing/reordered steps')
            for r in rows:
                domain='trackrad' if r['step']%2==0 else 'acdc'
                allowed=plan['ids']['train-40'] if domain=='trackrad' else plan['eligible']
                if r['domain']!=domain or r['sampling']['patient'] not in allowed: raise ValueError('Wrong training split/patient')
            paired=[{k:r[k] for k in keys} for r in rows]
            if reference is None: reference=paired; initial=info['initial_sidecar']
            if paired!=reference or initial!=info['initial_sidecar']: raise ValueError('Unpaired inputs/labels/initial sidecar')
            if arm in ('D1','G1'):
                if head_initial is None: head_initial=info['initial_head']
                if not head_initial or info['initial_head']!=head_initial: raise ValueError('Unpaired initial region head')
            if info['domain_counts']!={'acdc':proof['steps']//2,'trackrad':proof['steps']//2}: raise ValueError('Mix imbalance')
            files += [folder/'verified.json',folder/'proof.json',folder/'steps.json',folder/'final.pth']
    if phase=='smoke':
        if not require_smoke_inference: return arms,seeds,files
        for arm,view in [('M0','points'),('D1','region')]:
            folder=root/'smoke/validation-10/seed_0'/arm/view
            proof=cached(root,folder/'inference-verified.json')
            if not proof or len(proof['ids'])!=2: raise ValueError('Full deployment smoke inference required')
            if not cached(root,folder/'metrics-verified.json'): raise ValueError('Official smoke evaluation receipt missing')
            metrics=folder/'metrics.json'; official(metrics,proof['ids']); files += [metrics,folder/'inference-verified.json',folder/'metrics-verified.json']
    receipt(root,root/phase/'audit.json',files,phase=phase,arms=arms,seeds=seeds,paired=True,frozen_backbone_exact=True)
    print('AUDIT PASSED '+phase,flush=True)
    return arms,seeds,files

def official(path,cases):
    obj=read(path); rows=obj['results']; actual=[r['case_id'] for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=set(cases): raise ValueError('Official cases mismatch')
    agg={k:float(obj['aggregates'][v]) for k,v in FIELDS.items()}; agg['TimeSec']=float(obj['aggregates']['total_time'])
    if not all(math.isfinite(v) for v in agg.values()): raise ValueError('Nonfinite aggregate')
    for r in rows:
        if not all(math.isfinite(float(r[v])) for v in FIELDS.values()): raise ValueError('Nonfinite case metric')
    return agg,{r['case_id']:{k:float(r[v]) for k,v in FIELDS.items()} for r in rows}

def record_metrics(root,folder):
    checked(root)
    if cached(root,folder/'metrics-verified.json'): return
    proof=cached(root,folder/'inference-verified.json')
    if not proof: raise ValueError('Missing prediction receipt')
    official(folder/'metrics.json',proof['ids'])
    data=(folder/'evaluation.log').read_bytes()
    encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    payloads=[s.split('=',1)[1] for s in data.decode(encoding).splitlines() if s.startswith('REGION_METRICS_BASE64=')]
    if not payloads or base64.b64decode(payloads[-1],validate=True)!=(folder/'metrics.json').read_bytes():
        raise ValueError('Metrics bytes differ from official evaluator transport')
    receipt(root,folder/'metrics-verified.json',[folder/'metrics.json',folder/'inference-verified.json'],byte_identical=True)
    print('OFFICIAL METRICS VERIFIED '+str(folder),flush=True)

def test_gate(root):
    checked(root); training_audit(root,'full')
    p=root/'reports/full/validation-10'
    if not cached(root,p/'verified.json'): raise ValueError('Frozen full validation report required before test')
    d=read(p/'selection.json')
    expected=[(a,v) for a in expected_arms(root,'full') for v in arm_outputs(a)]
    if d['test_outputs']!=[list(x) for x in expected] or d['selection_split']!='validation-10': raise ValueError('Test plan changed')
    print('TEST GATE VERIFIED; public test is reporting only',flush=True)
    return d

def report(root,phase,split):
    plan=checked(root)
    if split=='test-38': test_gate(root)
    arms,seeds,files=training_audit(root,phase)
    rows=[]; summaries=[]; metrics={}; percase={}
    outputs=[('pretrained','points')]+[(a,v) for a in arms for v in arm_outputs(a)]
    for arm,view in outputs:
        group=[]
        for seed in ([0] if arm=='pretrained' else seeds):
            folder=root/phase/split/f'seed_{seed}'/arm/view
            inf=cached(root,folder/'inference-verified.json')
            if not inf or inf['ids']!=plan['ids'][split]: raise ValueError('Complete inference receipt required')
            if not cached(root,folder/'metrics-verified.json'): raise ValueError('Official evaluation proof required')
            agg,case=official(folder/'metrics.json',plan['ids'][split])
            row=dict(Split=split,Phase=phase,Arm=arm,Output=view,Seed=seed,**agg)
            rows.append(row); group.append(row); metrics[(arm,view,seed)]=agg; percase[(arm,view,seed)]=case
            files += [folder/'metrics.json',folder/'inference-verified.json',folder/'metrics-verified.json']
        summary=dict(Split=split,Phase=phase,Arm=arm,Output=view,N=len(group))
        for m in list(FIELDS)+['TimeSec']:
            summary[m+'_Mean']=statistics.mean(r[m] for r in group)
            summary[m+'_SD']=statistics.stdev(r[m] for r in group) if len(group)>1 else 0.
        summaries.append(summary)
        if arm=='pretrained':
            for seed in seeds:
                metrics[(arm,view,seed)]=metrics[(arm,view,0)]
                percase[(arm,view,seed)]=percase[(arm,view,0)]
    pairs=[((a,v),('M0','points')) for a,v in outputs if a not in ('pretrained','M0')]
    pairs += [(('D1','points'),('W1','points')),(('D1','region'),('D1','points'))]
    pairs += [((a,v),('pretrained','points')) for a,v in outputs if a!='pretrained']
    if 'G1' in arms: pairs += [(('G1','region'),('D1','region'))]
    deltas=[]; cases=[]
    for (a,v),(b,w) in pairs:
        for seed in seeds:
            prefix=dict(Split=split,Seed=seed,Candidate=a+'-'+v,Reference=b+'-'+w)
            d={m:metrics[a,v,seed][m]-metrics[b,w,seed][m] for m in FIELDS}
            deltas.append(dict(**prefix,**{'Delta_'+m:x for m,x in d.items()}))
            for case in plan['ids'][split]:
                cases.append(dict(**prefix,Case=case,**{'Delta_'+m:percase[a,v,seed][case][m]-percase[b,w,seed][case][m] for m in FIELDS}))
    def valid(a,v,b='M0',w='points'):
        return qualifies([{m:metrics[a,v,s][m]-metrics[b,w,s][m] for m in FIELDS} for s in seeds])
    folder=root/'reports'/phase/split
    for name,value in [('summary',summaries),('per-seed',rows),('paired-deltas',deltas),('case-deltas',cases)]:
        csv_save(folder/(name+'.csv'),value); files.append(folder/(name+'.csv'))
    if split=='validation-10':
        eligible=[(a,v) for a,v in outputs if a not in ('pretrained','M0') and valid(a,v)]
        rank=sorted(eligible,key=lambda x:(-statistics.mean(metrics[(x[0],x[1],s)]['DSC'] for s in seeds),x))
        chosen=rank[0] if rank else ('M0','points')
        decision=dict(selection_split=split,region_gate=valid('D1','region'),eligible=eligible,selected=chosen,
            test_outputs=[(a,v) for a in arms for v in arm_outputs(a)],rule=RULE['selection'],
            selected_qualifies_vs_pretrained=valid(chosen[0],chosen[1],'pretrained','points'),
            note='Mask performance only, not physical point identity; no clinical claim. Test is repeatedly inspected public data.')
        # JSON converts tuples to lists; normalize before frozen comparison.
        freeze(folder/'selection.json',json.loads(json.dumps(decision))); files.append(folder/'selection.json')
    receipt(root,folder/'verified.json',files,phase=phase,split=split)
    for s in summaries: print(f'RESULT {s["Arm"]}/{s["Output"]}: DSC={s["DSC_Mean"]:.6f}; D98={s["D98_Mean"]:.6f}',flush=True)
    print(f'REPORT COMPLETE {phase}/{split}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['audit','report','test-gate','record']); p.add_argument('--phase',default='pilot',choices=['smoke','pilot','full'])
    p.add_argument('--folder')
    p.add_argument('--split',default='validation-10',choices=['validation-10','test-38']); a=p.parse_args(); root=Path('/study')
    if a.command=='audit': training_audit(root,a.phase,require_smoke_inference=a.phase=='smoke')
    elif a.command=='report': report(root,a.phase,a.split)
    elif a.command=='test-gate': test_gate(root)
    else: record_metrics(root,safe(root,a.folder))
