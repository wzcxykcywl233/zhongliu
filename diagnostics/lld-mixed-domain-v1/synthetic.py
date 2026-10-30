"""Known 2D affine motion from ONE real lesion slice, never z/contrast as time."""
import math
import numpy as np
import torch
import torch.nn.functional as F

MOTION = dict(frames=16,rotation_degrees=8.,translation_fraction=.04,scale_delta=.04,
              generator='linear plus sinusoidal affine path; first transform identity',
              resampling='bilinear, align_corners=True; image border padding; mask zero padding',
              visibility='known transformed source point inside image; no biological occlusion label')


def matrices(draw,height,width):
    rng=np.random.default_rng(draw+71)
    angle,dx,dy,scale=rng.uniform(-1,1,4)
    # Smooth but non-periodic endpoint drift; no repeated frames or temporal padding.
    t=torch.linspace(0,1,MOTION['frames'],dtype=torch.float32)
    ramp=t+.15*torch.sin(2*math.pi*t)
    theta=ramp*(angle*MOTION['rotation_degrees']*math.pi/180)
    sizes=1+ramp*(scale*MOTION['scale_delta'])
    a=torch.eye(3).repeat(len(t),1,1)
    a[:,0,0]=sizes*theta.cos(); a[:,1,1]=a[:,0,0]
    a[:,0,1]=-sizes*theta.sin(); a[:,1,0]=sizes*theta.sin()
    center=torch.tensor([(width-1)/2,(height-1)/2])
    delta=ramp[:,None]*torch.tensor([dx*width,dy*height])*MOTION['translation_fraction']
    a[:,:2,2]=center+delta-torch.einsum('tij,j->ti',a[:,:2,:2],center)
    a[0]=torch.eye(3)
    return a


def warp(image,mask,queries,forward):
    """Image C,H,W; queries 1,N,(t,x,y); forward maps source pixels to frame pixels."""
    if image.ndim!=3 or mask.shape!=(1,)+image.shape[-2:] or queries.ndim!=3 or queries.shape[0]!=1:
        raise ValueError('Invalid slice/mask/query shape')
    if forward.ndim!=3 or forward.shape[1:]!=(3,3) or len(forward)<2:
        raise ValueError('Invalid transform sequence')
    if not torch.equal(forward[0],torch.eye(3)) or not bool((queries[...,0]==0).all()):
        raise ValueError('Query frame must be exact identity at t=0')
    if not all(torch.isfinite(x).all() for x in (image,mask,queries,forward)) or image.min()<0 or image.max()>255:
        raise ValueError('Finite float32 image in [0,255] required')
    _,h,w=image.shape
    if h<2 or w<2 or torch.linalg.det(forward).abs().min()<.1: raise ValueError('Degenerate warp')
    if not ((queries[...,1]>=0)&(queries[...,1]<=w-1)&(queries[...,2]>=0)&(queries[...,2]<=h-1)).all():
        raise ValueError('Query outside source image')
    y,x=torch.meshgrid(torch.arange(h,dtype=torch.float32),torch.arange(w,dtype=torch.float32),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1)
    inverse=torch.linalg.inv(forward)
    source=torch.einsum('tij,hwj->thwi',inverse,pixels)[...,:2]
    grid=source*torch.tensor([2/(w-1),2/(h-1)])-1
    frames=F.grid_sample(image[None].expand(len(forward),-1,-1,-1),grid,
                         align_corners=True,padding_mode='border').clamp(0,255)
    masks=F.grid_sample(mask[None].expand(len(forward),-1,-1,-1),grid,
                        align_corners=True,mode='nearest',padding_mode='zeros')
    frames[0]=image; masks[0]=mask
    xy=torch.cat((queries[0,:,1:],torch.ones(queries.shape[1],1)),-1)
    tracks=torch.einsum('tij,nj->tni',forward,xy)[...,:2][None]
    visible=((tracks[...,0]>=0)&(tracks[...,0]<=w-1)&(tracks[...,1]>=0)&(tracks[...,1]<=h-1)).float()
    tracks[:,0]=queries[...,1:]
    target=dict(tracks=tracks,visibility=visible,confidence=torch.ones_like(visible))
    return frames.contiguous(),masks.contiguous(),target
