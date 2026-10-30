"""Pinned official download and patient-grouped 3D audit. No model/GPU needed."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

REPO = 'wanglab/LLD-MMRI-MedSAM2'
REVISION = 'b7e8da56b267587689d8440e8298205f3fc4914e'
CARD = 'https://huggingface.co/datasets/' + REPO
TERMS = 'https://github.com/LMMMEng/LLD-MMRI-Dataset'
SEQUENCES = ('C+A','C+Delay','C+V','C-pre','DWI','InPhase','OutPhase','T2WI')
SPLIT_SALT = 'lld-medcotracker-patient-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def freeze(path,value):
    if Path(path).exists():
        if read(path)!=value: raise ValueError('Frozen input changed; use new directory: '+str(path))
    else: save(path,value)


def safe(root,name):
    if not isinstance(name,str) or '\\' in name or ':' in name: raise ValueError('Unsafe filename')
    p=(Path(root)/name).resolve()
    if Path(root).resolve() not in p.parents: raise ValueError('Escaping filename')
    return p


def identity(name):
    m=re.fullmatch(r'images/(MR-?\d+)_(\d+)_(.+)_0000\.nii\.gz',name)
    if not m or m[3] not in SEQUENCES: raise ValueError('Unknown official image name: '+name)
    return dict(patient=m[1].replace('-',''),source_id=m[1]+'_'+m[2],sequence=m[3],
                image=name,label='labels/'+name[len('images/'):-len('_0000.nii.gz')]+'.nii.gz')


def catalog(meta):
    if meta['sha']!=REVISION: raise ValueError('Official revision changed')
    items={r['rfilename']:r for r in meta['siblings']}
    rows=[identity(n) for n in sorted(items) if n.startswith('images/')]
    if len(rows)!=3984 or len({r['patient'] for r in rows})!=498:
        raise ValueError('Expected 3984 scans / 498 patient groups; inspect naming before training')
    for r in rows:
        if r['label'] not in items: raise ValueError('Missing paired label')
    if {r['label'] for r in rows}!={n for n in items if n.startswith('labels/')}:
        raise ValueError('Unpaired label inventory')
    files={}
    for name in sorted(items):
        if name=='.gitattributes': continue
        row=items[name]; safe(Path('/catalog'),name)
        lfs=row.get('lfs')
        if lfs and (lfs['size']!=row['size'] or not re.fullmatch('[0-9a-f]{64}',lfs['sha256'])):
            raise ValueError('Invalid LFS digest')
        files[name]=dict(size=row['size'],sha256=lfs['sha256'] if lfs else None,
                         git_blob=None if lfs else row['blobId'])
    return dict(repo=REPO,revision=REVISION,files=files,scans=rows)


def valid_file(path,entry):
    if not path.is_file() or path.stat().st_size!=entry['size']: return False
    if entry.get('sha256'): return sha(path)==entry['sha256']
    h=hashlib.sha1(('blob '+str(entry['size'])+'\0').encode())
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()==entry['git_blob']


def fetch_file(root,name,entry,opener=urllib.request.urlopen):
    path=safe(root,name); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not valid_file(path,entry): raise ValueError('Existing file corrupt; preserve and inspect: '+name)
        return
    part=path.with_name(path.name+'.partial')
    url=CARD+'/resolve/'+REVISION+'/'+urllib.parse.quote(name,safe='/')+'?download=true'
    for attempt in range(4):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset==entry['size'] and valid_file(part,entry): os.replace(part,path); return
            if offset>=entry['size']: offset=0
            req=urllib.request.Request(url,headers={'User-Agent':'MedCoTracker-research/1',
                        **({'Range':f'bytes={offset}-'} if offset else {})})
            with opener(req,timeout=90) as response:
                status=getattr(response,'status',200)
                if status==206:
                    cr=response.headers.get('Content-Range','')
                    if not cr.startswith(f'bytes {offset}-'): raise ValueError('Wrong resumed byte range')
                elif status==200: offset=0
                else: raise ValueError('Unexpected download status')
                with part.open('ab' if offset else 'wb') as f:
                    size=offset
                    while True:
                        block=response.read(2**20)
                        if not block: break
                        size+=len(block)
                        if size>entry['size']: raise ValueError('Response exceeds declared file size')
                        f.write(block)
                    f.flush(); os.fsync(f.fileno())
            if not valid_file(part,entry): raise ValueError('Downloaded checksum/size mismatch: '+name)
            os.replace(part,path); return
        except (OSError,ValueError) as exc:
            if attempt==3: raise
            print('DOWNLOAD RETRY '+name+' '+str(exc),flush=True); time.sleep(2*(attempt+1))


def download(root,accepted):
    if not accepted: raise ValueError('Explicit noncommercial/no-redistribution acceptance required')
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'research-terms.json',dict(accepted=True,scope='noncommercial research; no external redistribution; '
        'same-organization single-site internal use only',dataset_card=CARD,original_terms=TERMS,revision=REVISION))
    lock=root/'download-manifest.json'
    if lock.exists(): inventory=read(lock)
    else:
        url='https://huggingface.co/api/datasets/'+REPO+'/revision/'+REVISION+'?blobs=true'
        with urllib.request.urlopen(url,timeout=90) as f: inventory=catalog(json.load(f))
        freeze(lock,inventory)
    if inventory['revision']!=REVISION or inventory['repo']!=REPO: raise ValueError('Wrong dataset lock')
    for i,(name,entry) in enumerate(inventory['files'].items()):
        fetch_file(root,name,entry)
        if i%25==0: print(f'DOWNLOAD VERIFIED {i+1}/{len(inventory["files"])} {name}',flush=True)
    save(root/'download-complete.json',dict(complete=True,manifest_sha256=sha(lock),
        files=len(inventory['files']),bytes=sum(e['size'] for e in inventory['files'].values())))
    print('DOWNLOAD COMPLETE; no training started',flush=True)


def split_groups(rows):
    """Keep all sequences/lesions per patient together; join exact cross-patient duplicates."""
    patients=sorted({r['patient'] for r in rows}); parent={p:p for p in patients}
    def find(p):
        while parent[p]!=p: p=parent[p]
        return p
    hashes=defaultdict(set)
    for r in rows: hashes[r['decoded_image_sha256']].add(r['patient'])
    for group in hashes.values():
        values=sorted(group)
        for p in values[1:]: parent[find(p)]=find(values[0])
    groups=defaultdict(list)
    for p in patients: groups[find(p)].append(p)
    ordered=sorted(groups.values(),key=lambda ps:hashlib.sha256((SPLIT_SALT+'|'+','.join(ps)).encode()).hexdigest())
    if len(ordered)<5: raise ValueError('Insufficient independent patient groups')
    n=max(1,round(.2*len(ordered))); dev={p for g in ordered[:n] for p in g}
    return {p:('external-dev' if p in dev else 'external-train') for p in patients}, [sorted(v) for v in hashes.values() if len(v)>1]


def scan_arrays(image,label):
    import numpy as np
    import SimpleITK as sitk
    if image.GetDimension()!=3 or label.GetDimension()!=3: raise ValueError('Require 3D volumes, not cine')
    if image.GetSize()!=label.GetSize(): raise ValueError('Image/mask shape mismatch')
    for getter in ('GetSpacing','GetOrigin','GetDirection'):
        if not np.allclose(getattr(image,getter)(),getattr(label,getter)(),rtol=1e-5,atol=1e-4):
            raise ValueError('Image/mask geometry mismatch: '+getter)
    a=sitk.GetArrayFromImage(image).astype(np.float32); m=sitk.GetArrayFromImage(label)
    if not np.isfinite(a).all() or not np.isfinite(m).all() or (m<0).any() or not np.equal(m,np.rint(m)).all():
        raise ValueError('Invalid voxel/label values')
    if any(v<=0 or not np.isfinite(v) for v in image.GetSpacing()): raise ValueError('Invalid spacing')
    slices=[]
    for z in range(len(a)):
        if (m[z]>0).sum()>=16 and np.percentile(a[z],99)>np.percentile(a[z],1): slices.append(z)
    if not slices: raise ValueError('No nonconstant lesion slice with >=16 foreground pixels')
    h=hashlib.sha256(str(a.shape).encode()); h.update(a.astype('<f4').tobytes())
    return dict(shape=list(a.shape),spacing=list(image.GetSpacing()),usable_slices=slices,
        label_values=np.unique(m).tolist(),decoded_image_sha256=h.hexdigest(),
        foreground_voxels=int((m>0).sum()))


def audit(root,output):
    import SimpleITK as sitk
    inv=read(root/'download-manifest.json'); complete=read(root/'download-complete.json')
    if inv['revision']!=REVISION or not complete['complete'] or complete['manifest_sha256']!=sha(root/'download-manifest.json'):
        raise ValueError('Completed pinned download required')
    rows=[]; errors=[]; output.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(inv['scans']):
        try:
            digests={r[k]:sha(safe(root,r[k])) for k in ('image','label')}
            if any(not valid_file(safe(root,n),inv['files'][n]) for n in digests): raise ValueError('Official checksum mismatch')
            marker=output/'scans'/(Path(r['label']).name+'.json')
            source=dict(files=digests,audit_code=sha(Path(__file__)),revision=REVISION)
            if marker.exists():
                cached=read(marker)
                if cached['source']!=source: raise ValueError('Cached audit changed')
                row=cached['row']
            else:
                row=dict(**r,**scan_arrays(sitk.ReadImage(str(root/r['image'])),sitk.ReadImage(str(root/r['label']))),files=digests)
                save(marker,dict(source=source,row=row))
            rows.append(row)
        except Exception as exc: errors.append(dict(image=r['image'],error=str(exc)))
        if i%25==0: print(f'AUDIT {i+1}/{len(inv["scans"])} errors={len(errors)}',flush=True)
    save(output/'audit-errors.json',errors)
    if errors:
        save(output/'summary.json',dict(complete=True,training_ready=False,errors=len(errors),automatic_exclusions=0))
        raise ValueError('Structural audit failed; inspect audit-errors.json; no automatic exclusions')
    splits,duplicates=split_groups(rows)
    for r in rows: r['split']=splits[r['patient']]
    data=dict(schema=1,complete=True,training_ready=True,revision=REVISION,audit_code=sha(Path(__file__)),
        download_manifest_sha256=sha(root/'download-manifest.json'),terms_sha256=sha(root/'research-terms.json'),
        split_salt=SPLIT_SALT,patients=splits,scans=rows,exact_duplicate_patient_groups=duplicates,
        acquisition='3D multisequence MRI, NOT temporal cine; sequence/z never used as time',
        note='Structural checks only, not clinical mask-quality certification; masks human-in-loop MedSAM2. '
             'Patient IDs inferred conservatively from filename; no identity certification.')
    freeze(output/'data-manifest.json',data)
    save(output/'summary.json',dict(complete=True,training_ready=True,scans=len(rows),patients=len(splits),
        split_counts={s:sum(v==s for v in splits.values()) for s in ('external-train','external-dev')},
        cross_split_exact_duplicates=0,quality_certified=False,automatic_exclusions=0,training_started=False))
    print('AUDIT COMPLETE '+str(output/'summary.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['download','audit'])
    p.add_argument('--data',type=Path,default=Path('/data')); p.add_argument('--output',type=Path,default=Path('/output'))
    p.add_argument('--accept-research-terms',action='store_true'); a=p.parse_args()
    if a.stage=='download': download(a.data,a.accept_research_terms)
    else: audit(a.data,a.output)
