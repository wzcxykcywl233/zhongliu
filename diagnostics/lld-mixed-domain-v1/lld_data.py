"""Balanced patient sampling. Real TrackRAD cache and synthetic LLD targets kept distinct."""
import numpy as np
import torch
import protocol as P
from data import trackrad_sample,resize
from synthetic import matrices,warp,MOTION


def choose(plan,seed,step):
    domain='trackrad' if step%2==0 else 'lld_synthetic'
    names=plan['ids']['train-40'] if domain=='trackrad' else sorted(
        p for p,s in plan['external']['patients'].items() if s=='external-train')
    epoch,index=divmod(step//2,len(names))
    order=np.random.default_rng(P.C.RULE['seed_base']+seed+epoch*97).permutation(len(names))
    draw=P.C.RULE['seed_base']+seed*1000003+step*101
    return domain,names[int(order[index])],draw


def sample(plan,lld,trackrad,seed,step):
    from resources.seg_to_tap import convert_mask_to_points
    domain,patient,draw=choose(plan,seed,step)
    if domain=='trackrad':
        s=trackrad_sample(patient,trackrad,draw)
        s['queries']=convert_mask_to_points(s['masks'][:1],n_border_points=P.C.RULE['queries'])
    else:
        import SimpleITK as sitk
        rows=[r for r in plan['external']['scans'] if r['patient']==patient and r['split']=='external-train']
        if not rows or plan['external']['patients'][patient]!='external-train': raise ValueError('External dev access forbidden')
        rng=np.random.default_rng(draw); r=rows[int(rng.integers(len(rows)))]; z=r['usable_slices'][int(rng.integers(len(r['usable_slices'])))]
        P.C.verify(lld,r['files'])
        image=sitk.GetArrayFromImage(sitk.ReadImage(str(lld/r['image'])))[z].astype(np.float32)
        mask=(sitk.GetArrayFromImage(sitk.ReadImage(str(lld/r['label'])))[z]>0).astype(np.uint8)
        v,m=resize(image[None],mask[None]); q=convert_mask_to_points(m,n_border_points=P.C.RULE['queries'])
        # Spline interpolation can overshoot the image boundary; bind exact clipped query.
        raw=q.clone(); q[...,1].clamp_(0,v.shape[-1]-1); q[...,2].clamp_(0,v.shape[-2]-1)
        a=matrices(draw,*v.shape[-2:]); video,masks,label=warp(v[0],m[0],q,a)
        s=dict(video=video,masks=masks,queries=q,label=label,annotated=torch.ones(len(a),dtype=torch.bool),
            sampling=dict(patient=patient,source_image=r['image'],source_label=r['label'],sequence=r['sequence'],
                spatial_slice_z=z,synthetic_frames=len(a),time_is_synthetic=True,
                forward_source_to_frame=a.tolist(),query_clipped_values=int((raw!=q).sum()),
                target_origin='exact synthetic affine correspondence, NOT annotated biological motion'))
    if s['queries'].shape!=(1,P.C.RULE['queries'],3): raise ValueError('Invalid query count')
    s.update(domain=domain,draw=draw)
    return s
