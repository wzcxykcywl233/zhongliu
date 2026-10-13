"""Pure diagnostic helpers; no model or optimizer modifications."""
import csv
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import numpy as np

PROFILE='hierarchical_full_grid0_iterations2_memory_topk_diverse'
CASES=('B_006','A_028','A_012','A_010')
ROUTES=('deployment_raw','deployment_normalized','training_exact')
MASK='output/images/mri-linac-series-targets/output.mha'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    temp.replace(path)

def csv_save(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['no_records']
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)

def split_displacement(reference,candidate):
    if reference.shape!=candidate.shape or reference.ndim!=4 or reference.shape[0]!=1 or reference.shape[-1]!=2:
        raise ValueError('Expected matched 1,T,N,2 tracks')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all(): raise ValueError('Nonfinite tracks')
    delta=candidate-reference
    global_delta=delta.mean(axis=2,keepdims=True)
    local_delta=delta-global_delta
    return global_delta,local_delta

def native_xy(points,native_shape,training_transposed=False):
    # align_corners=False resize pixel-center mapping. Not a point-GT claim.
    h,w=native_shape
    result=np.empty_like(points)
    if training_transposed:
        result[...,0]=(points[...,1]+.5)*w/384-.5
        result[...,1]=(points[...,0]+.5)*h/512-.5
    else:
        result[...,0]=(points[...,0]+.5)*w/512-.5
        result[...,1]=(points[...,1]+.5)*h/384-.5
    return result

def dice(a,b):
    n=int(a.sum())+int(b.sum())
    return 2*int((a&b).sum())/n if n else None

def geometry(pred,truth):
    d=dice(pred,truth)
    result={'Dice':d,'TruthArea':int(truth.sum()),'PredictionArea':int(pred.sum()),
            'CentroidErrorPixels':None,'CentroidErrorX':None,'CentroidErrorY':None}
    if truth.any() and pred.any():
        y,x=np.argwhere(pred).mean(0)-np.argwhere(truth).mean(0)
        result.update(CentroidErrorPixels=float(np.hypot(x,y)),CentroidErrorX=float(x),CentroidErrorY=float(y))
    return result

def summarize_masks(masks,truth):
    if masks.shape!=truth.shape: raise ValueError('Unpaired diagnostic masks')
    rows=[{'Frame':t,**geometry(masks[t],truth[t])} for t in range(1,len(truth))]
    return rows,{k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                 if any(r[k] is not None for r in rows) else None for k in geometry(masks[0],truth[0])}

def input_manifest(training,study,dataset,weights=None):
    protocol=read(training/'frozen-run.json'); frozen=read(study/'frozen-study.json')
    if sha(training/'frozen-run.json')!=frozen['OldProtocolSHA256'].lower(): raise ValueError('Original protocol changed')
    audit=read(training/'train/training-audit.json')
    if (sha(training/'train/training-audit.json')!=frozen['OldTrainingAuditSHA256'].lower() or
        not audit['complete'] or audit['steps']!=1000 or audit['seeds']!=[0,1,2] or
        not audit['paired_teacher_case_query_video_sequences']):
        raise ValueError('Original training audit changed/incomplete')
    groups={s:{r['Case'] for r in protocol['Dataset'] if r['Split']==s} for s in ('train-40','validation-10','test-38')}
    if list(map(len,groups.values()))!=[40,10,38] or len(set.union(*groups.values()))!=88: raise ValueError('Invalid 40/10/38 split')
    if not set(CASES)<=groups['validation-10']: raise ValueError('Fixed probe cases not in validation-10')
    hashes={'training/frozen-run.json':sha(training/'frozen-run.json'),'training/train/training-audit.json':sha(training/'train/training-audit.json'),
            'study/frozen-study.json':sha(study/'frozen-study.json')}
    expected={}
    for row in protocol['Dataset']:
        if row['Split']!='validation-10': continue
        parts=PureWindowsPath(row['File']).parts
        ids=[i for i,p in enumerate(parts) if p==row['Case']]
        if len(ids)!=1: raise ValueError('Ambiguous original dataset path')
        path=Path(*parts[ids[0]:])
        if '..' in path.parts or path.is_absolute() or str(path) in expected: raise ValueError('Unsafe/duplicate dataset entry')
        expected[str(path)]=row['SHA256'].lower()
    actual={str(p.relative_to(dataset)):sha(p) for p in dataset.rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('Validation files changed since training protocol')
    hashes.update({'dataset/'+k:v for k,v in actual.items()})
    for seed in (0,1,2):
        for kind in ('mlp','mamba'):
            relative=f'seed_{seed}/{kind}/cotracker_three_final.pth';p=training/'train'/relative
            if sha(p)!=audit['final_checkpoints_sha256'][relative]: raise ValueError('Trained checkpoint changed')
            hashes['training/train/'+relative]=sha(p)
    p=training/'pretrained-frozen.pth';hashes['training/pretrained-frozen.pth']=sha(p)
    if weights:
        for row in protocol['Weights']:
            p=weights/row['Name']
            if sha(p)!=row['SHA256'].lower(): raise ValueError('Original teacher/pretrained weights changed')
            hashes['weights/'+row['Name']]=sha(p)
    for leaf in ('study-spec.json','local-runtime-images.json'):
        hashes['study/'+leaf]=sha(study/leaf)
    for kind in ('mlp','mamba'):
        leaf='train/seed_0/'+kind+'/meta.json'
        hashes['training/'+leaf]=sha(training/leaf)
    for case in CASES:
        folders=[study/'validation-10/pretrained'/PROFILE/'checkpoint/jobs'/case]
        folders += [study/'validation-10'/f'seed_{seed}'/(kind+'_full')/PROFILE/'checkpoint/jobs'/case
                    for seed in (0,1,2) for kind in ('mlp','mamba')]
        for folder in folders:
            for leaf in (MASK,'diagnostics.json','output/sidecar-tracks.npz','.complete'):
                p=folder/leaf
                hashes['study/'+p.relative_to(study).as_posix()]=sha(p)
    return protocol,hashes

def frozen_output(output,manifest):
    path=output/'manifest.json'
    if path.exists():
        if read(path)!=manifest: raise ValueError('Probe inputs/code/images changed; use a new OutputRoot')
    else:
        if any(p.name not in ('.lock',) for p in output.iterdir()): raise ValueError('Unrecorded existing output; use a new OutputRoot')
        save(path,manifest)

def check_input_hashes(hashes,training,study,dataset,weights=None):
    roots={'training':training,'study':study,'dataset':dataset,'weights':weights}
    for key,checksum in hashes.items():
        group,relative=key.split('/',1)
        if roots[group] is None or sha(roots[group]/relative)!=checksum: raise ValueError('Source changed during probe: '+key)
