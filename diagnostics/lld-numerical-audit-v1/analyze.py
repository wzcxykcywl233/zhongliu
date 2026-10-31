"""Read-only CPU audit of saved runtime evidence. Never changes a training gate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import torch

GROUPS = ('outputs','gradients','classifier_gradients','student','classifier')
PAIR_KEYS = ('step','video','queries','sampling','domain','slot','draw',
             'label_tracks','label_visibility','label_confidence','label_origin')
SIDE = 'updateformer.feature_sidecar.'
ENCODER = ('fnet.layer4.','fnet.conv2.','fnet.conv3.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def safe(root, relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe relative path')
    p = (root/relative).resolve()
    if root.resolve() not in p.parents: raise ValueError('Path escapes source root')
    return p


def verify(root, files):
    for name, digest in files.items():
        if sha(safe(root,name)) != digest: raise ValueError('Changed input: '+name)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def tensor_stats(a,b,atol,rtol):
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):
        raise ValueError('Expected tensors')
    if a.shape != b.shape or a.dtype != b.dtype: raise ValueError('Shape/dtype changed')
    if a.is_complex(): raise ValueError('Unexpected complex tensor')
    # Mirrors the original assert_close(a,b): b is its expected tensor. Report
    # relative-L2 against the original/reference separately; zero norm is null.
    aa, bb = a.reshape(-1), b.reshape(-1)
    bad = different = flips = 0
    diff_sq = ref_sq = max_abs = max_scaled = 0.
    worst_flat = worst_ratio_flat = 0
    for start in range(0,aa.numel(),2**18):
        x=aa[start:start+2**18].double(); y=bb[start:start+2**18].double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite evidence')
        d=(x-y).abs(); tolerance=atol+rtol*y.abs()
        bad+=int((~torch.isclose(aa[start:start+2**18],bb[start:start+2**18],atol=atol,rtol=rtol,equal_nan=False)).sum())
        different+=int((x!=y).sum())
        flips+=int(((x*y)<0).sum())
        diff_sq+=float(d.square().sum()); ref_sq+=float(x.square().sum())
        scaled=d/tolerance.clamp_min(1e-300)
        if d.numel() and float(d.max())>max_abs:
            max_abs=float(d.max()); worst_flat=start+int(d.argmax())
        if scaled.numel() and float(scaled.max())>max_scaled:
            max_scaled=float(scaled.max()); worst_ratio_flat=start+int(scaled.argmax())
    return dict(elements=aa.numel(),mismatched_elements=bad,different_elements=different,
                sign_flips=flips,max_abs_difference=max_abs,max_tolerance_ratio=max_scaled,
                diff_l2_squared=diff_sq,reference_l2_squared=ref_sq,
                relative_l2=math.sqrt(diff_sq/ref_sq) if ref_sq else None,
                worst_flat=worst_flat,worst_ratio_flat=worst_ratio_flat,exact=different==0)


def compare_all(reference,candidate,limits,step):
    rows=[]; summary={}
    if set(reference)!=set(GROUPS) or set(candidate)!=set(GROUPS):
        raise ValueError('Unexpected evidence schema')
    for group in GROUPS:
        a,b=reference[group],candidate[group]
        if isinstance(a,list):
            if not isinstance(b,list): raise ValueError('Output structure changed')
            a,b=dict(enumerate(a)),dict(enumerate(b))
        if a.keys()!=b.keys(): raise ValueError('Tensor keys changed: '+group)
        prefix='output' if group=='outputs' else ('grad' if 'gradient' in group else 'state')
        local=[]
        for key in a:
            stats=tensor_stats(a[key],b[key],limits[prefix+'_atol'],limits[prefix+'_rtol'])
            row=dict(step=step,group=group,tensor=str(key),**stats)
            rows.append(row); local.append(row)
        n=sum(r['elements'] for r in local)
        bad=sum(r['mismatched_elements'] for r in local)
        diff=sum(r['diff_l2_squared'] for r in local)
        ref=sum(r['reference_l2_squared'] for r in local)
        summary[group]=dict(tensors=len(local),elements=n,mismatched_elements=bad,
            mismatched_tensors=sum(r['mismatched_elements']>0 for r in local),
            mismatch_fraction=bad/n if n else 0.,
            max_abs_difference=max((r['max_abs_difference'] for r in local),default=0.),
            max_tolerance_ratio=max((r['max_tolerance_ratio'] for r in local),default=0.),
            relative_l2=math.sqrt(diff/ref) if ref else None,within_existing_tolerance=bad==0)
    return summary,rows


def adam_estimate(previous_weight, gradients, lr, decay):
    """Float64 scalar explanation, NOT a GPU-bitwise optimizer replay."""
    m=v=0.
    for g in gradients:
        m=.9*m+.1*g; v=.999*v+.001*g*g
    count=len(gradients)
    if not count: raise ValueError('No gradients')
    adaptive=-lr*(m/(1-.9**count))/(math.sqrt(v/(1-.999**count))+1e-8)
    decay_delta=-lr*decay*previous_weight
    return dict(predicted_weight=previous_weight+adaptive+decay_delta,
                adaptive_update=adaptive,weight_decay_update=decay_delta,
                first_moment=m,second_moment=v)


def trace_coordinate(name,index,original,candidate,lr,decay):
    if len(original)!=len(candidate) or len(original)<2: raise ValueError('Need a paired update history')
    trace=dict(tensor=name,flat_index=index,lr=lr,weight_decay=decay,
               note='Uses saved clipped gradients and float64 AdamW formula; not a GPU-bitwise replay.')
    for label,packets in (('original',original),('candidate',candidate)):
        if any(name not in p['gradients'] for p in packets):
            return dict(trace,available=False,reason='Gradient absent in part of the saved history')
        gradients=[float(p['gradients'][name].reshape(-1)[index]) for p in packets]
        prior=float(packets[-2]['student'][name].reshape(-1)[index])
        actual=float(packets[-1]['student'][name].reshape(-1)[index])
        estimate=adam_estimate(prior,gradients,lr,decay)
        trace[label]=dict(clipped_gradients=gradients,previous_weight=prior,actual_weight=actual,
                          actual_update=actual-prior,formula_residual=actual-estimate['predicted_weight'],**estimate)
    x,y=trace['original'],trace['candidate']
    trace.update(available=True,previous_weight_difference=y['previous_weight']-x['previous_weight'],
                 current_weight_difference=y['actual_weight']-x['actual_weight'],
                 incremental_update_difference=y['actual_update']-x['actual_update'],
                 predicted_update_difference=(y['adaptive_update']+y['weight_decay_update'])-(x['adaptive_update']+x['weight_decay_update']))
    return trace


def load_evidence(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def analyze(source,output,checkout,arm='F0'):
    torch.set_num_threads(2)
    source,output,checkout=map(lambda p:Path(p).resolve(),(source,output,checkout))
    if arm not in ('F0','Fprobe','F001','F010'): raise ValueError('Invalid arm')
    for p in (source,checkout):
        if p==output or p in output.parents or output in p.parents:
            raise ValueError('Output must not overlap inputs')
    manifest=read(source/'manifest.json')
    verify(checkout,manifest['original_sources']); verify(checkout,manifest['repair_sources'])
    original_dir=source/'original'/arm; candidate_dir=source/'lowmem'/arm
    receipt=read(original_dir/'verified.json')
    if (receipt.get('complete'),receipt.get('manifest'),receipt.get('mode'),receipt.get('arm')) != (True,sha(source/'manifest.json'),'original',arm):
        raise ValueError('Original reference lacks valid receipt')
    verify(original_dir,receipt['files'])
    failure=read(candidate_dir/'equivalence-failure.json'); step=failure['step']
    if failure.get('complete') is not False or failure['arm']!=arm or failure['mode']!='lowmem' or step not in (0,1,2):
        raise ValueError('Invalid failure identity')
    original_rows=read(original_dir/'steps.json')
    candidate_rows=read(candidate_dir/'steps.json') if (candidate_dir/'steps.json').exists() else []
    if [r['step'] for r in original_rows]!=[0,1,2] or [r['step'] for r in candidate_rows]!=list(range(step)):
        raise ValueError('Unexpected evidence history')
    candidate_rows=candidate_rows+[failure['runtime_record']]
    for i,c in enumerate(candidate_rows):
        if any(c[k]!=original_rows[i][k] for k in PAIR_KEYS):
            raise ValueError('Inputs/labels are not paired')
    files=[source/'manifest.json',source/'environment.json',original_dir/'verified.json',original_dir/'steps.json',
           candidate_dir/'equivalence-failure.json']
    if (candidate_dir/'steps.json').exists(): files.append(candidate_dir/'steps.json')
    files += [original_dir/f'step-{i}.pth' for i in range(step+1)]
    files += [candidate_dir/f'step-{i}.pth' for i in range(step)] + [candidate_dir/'failed-evidence.pth']
    provenance=dict(inputs={p.relative_to(source).as_posix():sha(p) for p in files},
                    analysis_sources={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.ps1','.txt')},arm=arm,step=step)
    if (output/'inputs.json').exists() and read(output/'inputs.json')!=provenance:
        raise ValueError('Analysis inputs changed; use a NEW OutputRoot')
    save(output/'inputs.json',provenance)
    originals=[load_evidence(original_dir/f'step-{i}.pth') for i in range(step+1)]
    candidates=[load_evidence(candidate_dir/f'step-{i}.pth') for i in range(step)] + [load_evidence(candidate_dir/'failed-evidence.pth')]
    summaries=[]; tensors=[]; timing=[]
    for i,(a,b) in enumerate(zip(originals,candidates)):
        summary,rows=compare_all(a,b,manifest['limits'],i)
        summaries.append(dict(step=i,groups=summary)); tensors+=rows
        x,y=original_rows[i],candidate_rows[i]
        timing.append(dict(step=i,domain=x['domain'],original_seconds=x['measured_seconds'],
            candidate_seconds=y['measured_seconds'],original_peak_gib=x['peak_allocated']/2**30,
            candidate_peak_gib=y['peak_allocated']/2**30))
    frozen=[r for r in tensors if r['group']=='student' and not r['tensor'].startswith((SIDE,)+ENCODER)]
    failed_weights=sorted((r for r in tensors if r['step']==step and r['group']=='student' and r['mismatched_elements']),
                          key=lambda r:r['max_tolerance_ratio'],reverse=True)
    traces=[]
    if step>0:
        for r in failed_weights[:10]:
            traces.append(trace_coordinate(r['tensor'],r['worst_ratio_flat'],originals,candidates,
                          5e-5 if r['tensor'].startswith(SIDE) else 5e-6,1e-5))
    save(output/'update-traces.json',traces)
    with (output/'tensor-differences.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(tensors[0])); writer.writeheader(); writer.writerows(tensors)
    # Recheck all consumed files to detect concurrent source changes during analysis.
    verify(source,provenance['inputs'])
    result=dict(complete=True,arm=arm,failed_step=step,original_reference_receipt_verified=True,
        candidate_complete_receipt=False,existing_tolerances=manifest['limits'],comparisons=summaries,
        frozen_student_exact=all(r['exact'] for r in frozen),matched_step_runtime=timing,
        worst_failed_weight_tensors=[{k:r[k] for k in ('tensor','mismatched_elements','elements','max_abs_difference','max_tolerance_ratio','relative_l2','worst_flat','worst_ratio_flat')} for r in failed_weights[:10]],
        formula_trace_count=len(traces),training_authorized=False,gate_changed=False,
        note='Saved independent optimizer trajectories, not same-prestate recomputation equivalence. Small differences alone do not prove benign rounding or permit training. No GPU, model forward/backward, optimizer execution, data/test access or gate edits.')
    save(output/'summary.json',result)
    for group,s in summaries[-1]['groups'].items():
        print(f'NUMERICS {group}: mismatch={s["mismatched_elements"]}/{s["elements"]}; tensors={s["mismatched_tensors"]}; max_abs={s["max_abs_difference"]:.9g}',flush=True)
    print('NUMERICAL AUDIT COMPLETE; training remains blocked. '+str(output/'summary.json'),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,default=Path('/evidence'))
    p.add_argument('--output',type=Path,default=Path('/output')); p.add_argument('--checkout',type=Path,default=Path('/checkout'))
    p.add_argument('--arm',default='F0',choices=['F0','Fprobe','F001','F010']); a=p.parse_args()
    analyze(a.source,a.output,a.checkout,a.arm)
