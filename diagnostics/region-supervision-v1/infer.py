"""Deployment point pipeline; optional region decoder replaces ONLY reconstruction."""
import argparse
from datetime import datetime,timezone
import importlib.util
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from core import *
from region import RegionHead
from train import features,seed_all

def scalar(path,kind):
    v=read(path)
    if isinstance(v,dict) and set(v)=={'value'}: v=v['value']
    if kind=='number':
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError('Invalid metadata')
    elif not isinstance(v,str) or not v: raise ValueError('Invalid region')
    return v

def run_algorithm(m,head,frames,query,metadata,view):
    import resources
    spec=importlib.util.spec_from_file_location('region_deployment',APP/'model.py')
    deployment=importlib.util.module_from_spec(spec); spec.loader.exec_module(deployment)
    setup=resources.setup_model; convert=resources.convert_point_trajectory_to_mask_sequence
    previous=os.environ.get('COTRACKER_EXPERIMENT'); os.environ['COTRACKER_EXPERIMENT']=PROFILE
    resources.setup_model=lambda **kwargs:m
    used=[]
    if view=='region':
        if head is None: raise ValueError('Missing region head')
        video=torch.from_numpy(frames.transpose(2,1,0).copy()).float()[:,None]
        video=F.interpolate(video,size=RULE['resize'],mode='bilinear',align_corners=False).expand(-1,3,-1,-1)
        first=torch.from_numpy(query[:,:,0].T.copy()).float()[None,None]
        first=F.interpolate(first,size=RULE['resize'],mode='nearest')[0].cuda()
        def region_convert(trajectories,**kwargs):
            if used or trajectories.shape[0]!=1: raise ValueError('Unexpected decoder calls/batch')
            used.append(True); tracks=trajectories[0].cuda()
            qfeat=features(m,video[:1].cuda()); output=[]
            for i in range(0,len(video),4):
                logits=head(features(m,video[i:i+4].cuda()),first,tracks[i:i+4],qfeat,tracks[0])
                masks=(F.interpolate(logits,size=RULE['resize'],mode='bilinear',align_corners=False)>0).cpu()
                output.append(masks)
            return torch.cat(output)[None]  # B,T,1,H,W
        resources.convert_point_trajectory_to_mask_sequence=region_convert
    try:
        with torch.no_grad(): result=deployment.run_algorithm(frames,query,**metadata)
        if view=='region' and not used: raise ValueError('Region output path was bypassed')
        return result
    finally:
        resources.setup_model=setup; resources.convert_point_trajectory_to_mask_sequence=convert
        if previous is None: os.environ.pop('COTRACKER_EXPERIMENT',None)
        else: os.environ['COTRACKER_EXPERIMENT']=previous

def prediction(case,metadata,start,end):
    return dict(pk='jobs/'+case,status='Succeeded',started_at=start,completed_at=end,
        inputs=[{'interface':{'slug':slug},'value':metadata[key]} for slug,key in
                [('frame-rate','frame_rate'),('magnetic-field-strength','magnetic_field_strength'),('scanned-region','scanned_region')]]+
            [{'interface':{'slug':'mri-linac-series'},'image':{'name':case+'_frames.mha'}},
             {'interface':{'slug':'mri-linac-target'},'image':{'name':'target.mha'}}],
        outputs=[{'interface':{'slug':'mri-linac-series-targets','relative_path':'images/mri-linac-series-targets'},'image':{'name':'output.mha'}}])

def infer(root,data,weights,phase,split,seed,arm,view,smoke_case=False):
    from reports import test_gate,training_audit
    import SimpleITK as sitk
    from resources.model import setup_model
    from cotracker.models.core.cotracker.feature_sidecar import configure_sidecar_runtime
    plan=checked(root)
    if split=='test-38': test_gate(root)
    if phase=='smoke':
        if not smoke_case or split!='validation-10' or seed!=0: raise ValueError('Invalid smoke inference')
    else: training_audit(root,phase)
    if arm!='pretrained' and view not in arm_outputs(arm): raise ValueError('Invalid output view')
    if arm=='pretrained' and (seed!=0 or view!='points'): raise ValueError('Reference is deterministic single output')
    verify(data,plan['files'][split]); verify(weights,plan['weights'])
    checkpoint=weights/'scaled_offline.pth' if arm=='pretrained' else root/phase/f'train/seed_{seed}'/arm/'final.pth'
    if arm!='pretrained' and not cached(root,checkpoint.parent/'verified.json'): raise ValueError('Training receipt missing')
    folder=root/phase/split/f'seed_{seed}'/arm/view
    ids=plan['ids'][split][:2] if smoke_case else plan['ids'][split]
    freeze(folder/'inputs.json',dict(manifest=sha(root/'manifest.json'),checkpoint=sha(checkpoint),ids=ids,view=view,profile=PROFILE,
                                   phase=phase,split=split,seed=seed,arm=arm))
    m=setup_model(str(checkpoint),device='cuda').requires_grad_(False)
    if arm!='pretrained': configure_sidecar_runtime(m,backend='compact')
    h=None
    if view=='region':
        h=RegionHead().cuda().eval(); h.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['region_head'])
    predictions=[]; artifacts=[folder/'inputs.json']
    for case in ids:
        job=folder/'jobs'/case; marker=job/'verified.json'
        files=[job/'prediction.json',job/'output/images/mri-linac-series-targets/output.mha']
        if cached(root,marker):
            predictions.append(read(files[0])); artifacts+=files+[marker]; continue
        casepath=data/case
        image=sitk.ReadImage(str(casepath/'images'/f'{case}_frames.mha'))
        # No future mask is ever loaded by inference.
        query_path=casepath/'targets'/(case+'_first_label.mha')
        if not query_path.is_file(): raise FileNotFoundError(query_path)
        array=sitk.GetArrayFromImage(image).astype(np.float32)
        query=sitk.GetArrayFromImage(sitk.ReadImage(str(query_path)))
        if query.ndim!=3 or query.shape!=array.shape[:2]+(1,): raise ValueError('First mask shape')
        meta=dict(frame_rate=scalar(casepath/'frame-rate.json','number'),magnetic_field_strength=scalar(casepath/'b-field-strength.json','number'),
                  scanned_region=scalar(casepath/'scanned-region.json','string'))
        seed_all(RULE['seed_base']); start=datetime.now(timezone.utc).isoformat()
        output=run_algorithm(m,h,array,query,meta,view)
        torch.cuda.synchronize(); end=datetime.now(timezone.utc).isoformat()
        if output.shape!=array.shape or not np.isfinite(output).all() or not np.isin(output,[0,1]).all():
            raise ValueError('Invalid prediction mask')
        files[1].parent.mkdir(parents=True,exist_ok=True)
        im=sitk.GetImageFromArray(output.astype(np.uint8)); im.CopyInformation(image)
        sitk.WriteImage(im,str(files[1]),True)
        if not np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(files[1]))),output): raise ValueError('MHA roundtrip failed')
        pred=prediction(case,meta,start,end); save(files[0],pred)
        receipt(root,marker,files,case=case,checkpoint=sha(checkpoint),view=view)
        artifacts+=files+[marker]; predictions.append(pred)
        print(f'INFER {phase}/{split} seed={seed} {arm}/{view} {case}',flush=True)
    save(folder/'predictions.json',predictions); artifacts.append(folder/'predictions.json')
    receipt(root,folder/'inference-verified.json',artifacts,ids=ids,phase=phase,split=split,seed=seed,arm=arm,view=view)
    print(f'INFERENCE COMPLETE {phase}/{split} {arm}/{view}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['smoke','pilot','full'],required=True)
    p.add_argument('--split',choices=['validation-10','test-38'],default='validation-10')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--arm',choices=['pretrained']+list(ARMS),required=True)
    p.add_argument('--view',choices=['points','region'],required=True); p.add_argument('--smoke-case',action='store_true')
    a=p.parse_args(); infer(Path('/study'),Path('/data'),Path('/weights'),a.phase,a.split,a.seed,a.arm,a.view,a.smoke_case)
