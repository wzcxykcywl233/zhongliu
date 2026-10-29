"""Audited one-line adapter for v1; original training sources/manifest stay frozen."""
import argparse
import hashlib
from pathlib import Path
import sys
import types
import torch

PACKAGE=Path(__file__).resolve().parent
ORIGINAL=PACKAGE.parent/'region-supervision-v1'
sys.path.insert(0,str(ORIGINAL))
import core

ORIGINAL_INFER_SHA256='83d10ec7643a8db5b13818bd9726606ac3e0984aa5d05e3901b0ecd8bae27688'
BINDING='region-output-shape-fix-v1.json'
BEFORE='return torch.cat(output)[None]  # B,T,1,H,W'
AFTER='return join_region_masks(output)  # B,T,H,W; audited singleton-channel repair'

def join_region_masks(chunks):
    """The native reconstruction/reshape contract is B,T,H,W, including T=1."""
    if not chunks: raise ValueError('Empty region output')
    spatial=chunks[0].shape[-2:]
    if any(x.ndim!=4 or x.shape[0]<1 or x.shape[1]!=1 or x.dtype!=torch.bool or x.shape[-2:]!=spatial for x in chunks):
        raise ValueError('Expected region masks shaped T,1,H,W with bool dtype')
    return torch.cat(chunks,dim=0)[:,0].unsqueeze(0)

def load_fixed():
    path=ORIGINAL/'infer.py'
    source=path.read_text(encoding='utf-8-sig')  # universal newline normalization
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_INFER_SHA256 or source.count(BEFORE)!=1:
        raise ValueError('Original inference changed; do not apply this repair to another version')
    module=types.ModuleType('region_verified_shape_repair')
    module.__file__=str(path)
    module.join_region_masks=join_region_masks
    exec(compile(source.replace(BEFORE,AFTER),str(path),'exec'),module.__dict__)
    return module

def bind(root):
    core.checked(root)  # strict old protocol, weights/source references and manifest retained
    source={p.name:core.sha(p) for p in sorted(PACKAGE.iterdir()) if p.is_file() and p.suffix in ('.py','.ps1','.txt')}
    value=dict(schema=1,manifest_sha256=core.sha(root/'manifest.json'),source=source,
               original_infer_sha256=ORIGINAL_INFER_SHA256,change={'before':BEFORE,'after':AFTER},
               scope='Only region inference singleton channel; no training/loss/threshold/metric change')
    path=root/BINDING
    core.freeze(path,value)
    return path

def run(root,data,weights,phase,split,seed,arm,view,smoke_case=False):
    module=load_fixed(); binding=bind(root)
    if view=='region':
        folder=root/phase/split/f'seed_{seed}'/arm/view
        # Never silently adopt any successful legacy region output from another implementation.
        proofs=list((folder/'jobs').glob('*/verified.json'))
        if (folder/'inference-verified.json').exists(): proofs.append(folder/'inference-verified.json')
        for p in proofs:
            proof=core.cached(root,p)
            if proof['files'].get(BINDING)!=core.sha(binding): raise ValueError('Region cache lacks this repair provenance')
        original_receipt=module.receipt
        def receipt(root,path,files,**extra):
            return original_receipt(root,path,list(files)+[binding],**extra)
        module.receipt=receipt
    # Point-only runs and receipts are byte-for-byte on the old code path; preserve M0 metrics bindings.
    return module.infer(root,data,weights,phase,split,seed,arm,view,smoke_case)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['smoke','pilot','full'],required=True)
    p.add_argument('--split',choices=['validation-10','test-38'],default='validation-10')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--arm',choices=['pretrained']+list(core.ARMS),required=True)
    p.add_argument('--view',choices=['points','region'],required=True); p.add_argument('--smoke-case',action='store_true')
    a=p.parse_args()
    run(Path('/study'),Path('/data'),Path('/weights'),a.phase,a.split,a.seed,a.arm,a.view,a.smoke_case)
