"""Never invent intermediate masks or repeat frames. Initial mask is a prompt."""
import numpy as np
import torch
import torch.nn.functional as F
from core import *

def resize(video, masks):
    lo,hi=np.percentile(video,[1,99]).astype(np.float32)
    if hi<=lo: raise ValueError('Constant clip')
    v=torch.from_numpy(np.ascontiguousarray(np.clip((video-lo)/(hi-lo),0,1)*np.float32(255)))
    v=F.interpolate(v[:,None],size=RULE['resize'],mode='bilinear',align_corners=False).clamp(0,255).expand(-1,3,-1,-1).contiguous()
    m=F.interpolate(torch.from_numpy(np.ascontiguousarray(masks)).float()[:,None],size=RULE['resize'],mode='nearest')
    return v,m

def external_sample(row, root, draw):
    import SimpleITK as sitk
    folder=root/'training'/row['patient']
    cine=sitk.GetArrayFromImage(sitk.ReadImage(str(root/row['cine']))).astype(np.float32)
    ed,es=(int(row['annotation_frames_zero_based'][p]) for p in ('ED','ES'))
    if ed==es: raise ValueError('ED=ES cannot supervise motion')
    phases=[ed,es] if draw%2==0 else [es,ed]
    direction=1 if phases[1]>phases[0] else -1
    indices=list(range(phases[0],phases[1]+direction,direction))
    labels=[sitk.GetArrayFromImage(sitk.ReadImage(str(folder/f"{row['patient']}_frame{t+1:02d}_gt.nii.gz"))) for t in phases]
    # Sampling uses QUERY mask only, never selects a slice for target-frame quality.
    choices=[(z,c) for z in range(labels[0].shape[0]) for c in (1,2,3) if (labels[0][z]==c).sum()>=16]
    if not choices: raise ValueError('No valid query foreground')
    rng=np.random.default_rng(draw); z,c=choices[int(rng.integers(len(choices)))]
    mask=np.zeros((len(indices),)+cine.shape[-2:],dtype=np.uint8)
    mask[0]=labels[0][z]==c; mask[-1]=labels[1][z]==c
    video,mask=resize(cine[indices,z],mask)
    annotated=torch.zeros(len(indices),dtype=torch.bool); annotated[[0,-1]]=True
    return dict(video=video,masks=mask,annotated=annotated,
        sampling=dict(patient=row['patient'],slice_z=z,category=c,native_frames=indices,annotated_indices=[0,len(indices)-1]))

def trackrad_sample(case, root, draw):
    import SimpleITK as sitk
    array=sitk.GetArrayFromImage(sitk.ReadImage(str(root/case/'images'/f'{case}_frames.mha'))).astype(np.float32)
    mask=sitk.GetArrayFromImage(sitk.ReadImage(str(root/case/'targets'/f'{case}_labels.mha')))
    if array.shape!=mask.shape or array.ndim!=3: raise ValueError('TrackRAD shape mismatch')
    # W,H,T -> T,H,W, matching deployment model.py (not historical loader's transposed convention).
    video=array.transpose(2,1,0); masks=(mask>0).transpose(2,1,0).astype(np.uint8)
    frames=min(RULE['trackrad_frames'],len(video))
    starts=[t for t in range(len(video)-frames+1) if masks[t].sum()>=16]
    if not starts or frames<2: raise ValueError('No usable native TrackRAD clip: '+case)
    start=starts[int(np.random.default_rng(draw).integers(len(starts)))]
    v,m=resize(video[start:start+frames],masks[start:start+frames])
    return dict(video=v,masks=m,annotated=torch.ones(frames,dtype=torch.bool),
        sampling=dict(patient=case,native_frames=list(range(start,start+frames)),category=1))

def sample(plan, audit, acdc, trackrad, seed, step):
    draw=RULE['seed_base']+seed*1000003+step*101
    domain='trackrad' if step%2==0 else 'acdc'
    names=plan['ids']['train-40'] if domain=='trackrad' else plan['eligible']
    k=step//2; epoch,index=divmod(k,len(names))
    order=np.random.default_rng(RULE['seed_base']+seed+epoch*97).permutation(len(names))
    name=names[int(order[index])]
    if domain=='trackrad': result=trackrad_sample(name,trackrad,draw)
    else:
        row=next(r for r in audit['patients'] if r['patient']==name)
        result=external_sample(row,acdc,draw)
    from resources.seg_to_tap import convert_mask_to_points
    result['queries']=convert_mask_to_points(result['masks'][0:1],n_border_points=RULE['queries'])
    if result['queries'].shape!=(1,RULE['queries'],3): raise ValueError('Contour query shape')
    result.update(domain=domain,draw=draw)
    return result
