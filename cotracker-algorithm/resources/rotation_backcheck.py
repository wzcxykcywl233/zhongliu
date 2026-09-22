"""History-guided local rotation, observation only; no labels or state feedback."""
import math
import torch
import torch.nn.functional as F
try:
    from .frame_backcheck import sample, inside
except ImportError:
    from frame_backcheck import sample, inside

ANGLES = (-20., -10., 0., 10., 20.)
PATCH = 64
MAX_POINTS = 16


def rotate(offset, angle):
    radians = math.radians(angle)
    x,y = offset.unbind(-1)
    return torch.stack((math.cos(radians)*x-math.sin(radians)*y,
                        math.sin(radians)*x+math.cos(radians)*y),-1)


def decide(scores):
    """Hard fallback to zero on ambiguous peaks; thresholds frozen, not fitted."""
    top = scores.topk(2,dim=-1)
    gain = top.values[...,0]-scores[...,2]
    gap = top.values[...,0]-top.values[...,1]
    accept = (gain >= .01) & (gap >= .02) & (top.indices[...,0] != 2)
    chosen = torch.where(accept,top.indices[...,0],2)
    return chosen,accept,gain,gap


def crop(image, centers, angle):
    """Rotate source content +angle: output s samples source Q+R(-angle)s.

    Pixel y increases downwards, so positive angles are clockwise on screen.
    No crop clipping: report invalid whole crops, require all angles in bounds.
    """
    h,w = image.shape[-2:]
    axis = torch.arange(PATCH,device=image.device,dtype=torch.float32)-(PATCH-1)/2
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    offsets = rotate(torch.stack((xx,yy),-1),-angle)
    xy = centers[:,None,None,:]+offsets
    valid = ((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).all(dim=(-2,-1))
    grid = torch.stack((2*xy[...,0]/(w-1)-1,2*xy[...,1]/(h-1)-1),-1)
    patches = F.grid_sample(image[None].float().expand(len(centers),-1,-1,-1),grid,
                            align_corners=True,padding_mode='zeros')
    return patches,valid


def encode_centers(model, patches):
    vectors=[]
    for chunk in patches.split(4):
        # Exact input convention of CoTracker's full-frame encoder.
        fmap = model.fnet((2*(chunk/255.)-1).expand(-1,3,-1,-1))
        fmap = F.normalize(fmap.float(),dim=1,eps=1e-6)
        xy = fmap.new_full((len(chunk),1,2),(PATCH-1)/2)
        vectors.append(sample(fmap,xy,model.stride)[:,0])
    return torch.cat(vectors)


@torch.no_grad()
def score_rotation_frames(model,video,features,queries,final,history,start):
    if model.training:
        raise ValueError('rotation observation requires eval mode (no BN updates)')
    if len(history)!=2 or final.shape[0]!=1 or features.shape[:2]!=final.shape[:2]:
        raise ValueError('rotation diagnostic requires batch1, two iterations and aligned features')
    device=features.device
    video=video.to(device=device,dtype=torch.float32)
    ids=torch.linspace(0,final.shape[2]-1,min(MAX_POINTS,final.shape[2]),device=device).round().long()
    q=queries.to(device)[0,ids,1:3].float()
    p=final.to(device)[0,:,ids].float()
    prior=history[0].to(device)[0,:,ids].float()
    if not all(bool(torch.isfinite(x).all()) for x in (video,features,q,p,prior)):
        raise ValueError('nonfinite rotation input')
    qvectors=[]
    qvalid=torch.ones(len(ids),device=device,dtype=torch.bool)
    # Cache five angle crops per segment, bounded at 16 points x 5 angles.
    for angle in ANGLES:
        patches,valid=crop(video[0,0],q,angle)
        qvectors.append(encode_centers(model,patches))
        qvalid &= valid
    qvectors=torch.stack(qvectors,dim=1)
    rows=[]
    def avg(x):
        return float(x.mean().item()) if x.numel() else None
    for t in range(1,len(p)):
        delta=prior[t]-p[t]
        radius=delta.norm(dim=-1)
        supported=qvalid & (radius>=.5) & (radius<=16)
        patches,tvalid=crop(video[0,t],p[t],0)
        supported &= tvalid
        target=features[:,t].float()
        reference=features[:,0].float()
        supported &= inside(prior[t][None],target,model.stride)[0]
        target_probe=sample(target,prior[t][None],model.stride)[0]
        scores=[]
        for angle in ANGLES:
            qprobe=q+rotate(delta,-angle)
            supported &= inside(qprobe[None],reference,model.stride)[0]
            scores.append((sample(reference,qprobe[None],model.stride)[0]*target_probe).sum(-1).clamp(-1,1))
        scores=torch.stack(scores,dim=-1)
        chosen,accept,gain,gap=decide(scores)
        target_vector=encode_centers(model,patches)
        zero=(qvectors[:,2]*target_vector).sum(-1).clamp(-1,1)
        aligned=(qvectors[torch.arange(len(ids),device=device),chosen]*target_vector).sum(-1).clamp(-1,1)
        original=(sample(reference,q[None],model.stride)[0]*sample(target,p[t][None],model.stride)[0]).sum(-1).clamp(-1,1)
        # Center was NOT used to select the angle: post-encoding center is
        # held out spatially from the historical offset used in coarse search.
        row=dict(frame=start+t,query_frame=start,points=len(ids),supported_points=int(supported.sum()),
            full_point_count=final.shape[2],sampled_point_indices=ids.tolist(),
            angle_accepted_points=int((accept&supported).sum()),
            angle_fallback_points=int((~accept&supported).sum()),
            crop_rejected_points=int((~(qvalid&tvalid)).sum()),
            radius_rejected_points=int(((radius<.5)|(radius>16)).sum()),
            mean_radius=avg(radius[supported]),coarse_gain=avg(gain[supported]),coarse_gap=avg(gap[supported]),
            center=avg(original[supported]),patch_zero=avg(zero[supported]),patch_aligned=avg(aligned[supported]),
            center_gain_aligned_minus_zero=avg((aligned-zero)[supported]))
        for i,angle in enumerate(ANGLES):
            row[f'angle_{int(angle)}_points']=int(((chosen==i)&supported).sum())
            row[f'coarse_score_{int(angle)}']=avg(scores[:,i][supported])
        rows.append(row)
    return rows
