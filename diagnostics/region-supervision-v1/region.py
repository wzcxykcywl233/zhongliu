"""Query-conditioned region residual. No future masks or teacher coordinates as inputs."""
import torch
from torch import nn
import torch.nn.functional as F

def prior_and_heat(first, tracks, size, query_tracks=None):
    """Differentiable translation prior + contour occupancy; no track-index GT."""
    t,n,_=tracks.shape; h,w=first.shape[-2:]; oh,ow=size
    small=F.interpolate(first.reshape(1,1,h,w).float(),size=size,mode='bilinear',align_corners=False)
    shift=tracks.mean(1)-(tracks[0] if query_tracks is None else query_tracks).mean(0)
    theta=torch.eye(2,3,device=tracks.device,dtype=tracks.dtype)[None].repeat(t,1,1)
    theta[:,:,2]=-2*shift/torch.tensor([w,h],device=tracks.device)
    grid=F.affine_grid(theta,(t,1,oh,ow),align_corners=False)
    prior=F.grid_sample(small.expand(t,-1,-1,-1),grid,align_corners=False)
    yy,xx=torch.meshgrid(torch.arange(oh,device=tracks.device),torch.arange(ow,device=tracks.device),indexing='ij')
    heat=torch.zeros(t,oh,ow,device=tracks.device)
    for chunk in tracks.split(32,dim=1):
        x=(chunk[...,0]+.5)*ow/w-.5; y=(chunk[...,1]+.5)*oh/h-.5
        d=(xx[None,None]-x[:,:,None,None]).square()+(yy[None,None]-y[:,:,None,None]).square()
        heat=heat+torch.exp(-d/8).sum(1)/n
    return prior,heat[:,None]*20

class RegionHead(nn.Module):
    def __init__(self,channels=128):
        super().__init__()
        self.project=nn.Conv2d(channels,32,1)
        self.decode=nn.Sequential(nn.Conv2d(36,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                                  nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.decode[-1].weight); nn.init.zeros_(self.decode[-1].bias)

    def forward(self,features,first,tracks,query_feature,query_tracks=None):
        size=features.shape[-2:]
        mask=F.interpolate(first.reshape(1,1,*first.shape[-2:]).float(),size=size,mode='nearest')
        proto=(query_feature*mask).sum((-1,-2))/mask.sum().clamp_min(1)
        similarity=F.cosine_similarity(features,proto[:,:,None,None],dim=1)[:,None]
        prior,heat=prior_and_heat(first,tracks,size,query_tracks)
        diff=features-query_feature
        change=diff.square().mean(1,keepdim=True).sqrt()
        residual=self.decode(torch.cat([self.project(features),prior,heat,similarity,change],1))
        # A bounded initial prior, not an irreversible hard mask.
        return torch.logit(prior.clamp(.05,.95))+residual

def region_loss(logits,masks,annotated):
    eligible=annotated.clone(); eligible[0]=False
    if not bool(eligible.any()): return logits.sum()*0
    y=F.interpolate(masks[eligible].float(),size=logits.shape[-2:],mode='nearest')
    x=logits[eligible]; p=x.sigmoid()
    dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
    return F.binary_cross_entropy_with_logits(x,y)+dice.mean()

def boundary_loss(tracks,masks,annotated,visible,distance_scale=8.):
    """Query points are contour points. Distance to contour, NOT point correspondences.

    Missing/empty annotated targets and invisible tracks are excluded; no label-derived
    visibility supervision is invented. Distances are in model pixels / 8.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion,distance_transform_edt
    t,n,_=tracks.shape; h,w=masks.shape[-2:]; maps=[]; valid=[]
    for i in range(t):
        m=masks[i,0].detach().cpu().numpy()>.5
        enabled=bool(annotated[i]) and i!=0 and bool(m.any())
        edge=m & ~binary_erosion(m)
        dist=distance_transform_edt(~edge) if enabled else np.zeros_like(m,dtype=np.float32)
        maps.append(dist.astype(np.float32)); valid.append(enabled)
    field=torch.as_tensor(np.stack(maps),device=tracks.device)[:,None]/distance_scale
    grid=tracks.clone(); grid[...,0]=2*grid[...,0]/max(w-1,1)-1; grid[...,1]=2*grid[...,1]/max(h-1,1)-1
    value=F.grid_sample(field,grid[:,:,None],padding_mode='border',align_corners=True)[:,0,:,0]
    outside=(tracks[...,0].clamp_max(0).abs()+(tracks[...,0]-(w-1)).clamp_min(0)+
             tracks[...,1].clamp_max(0).abs()+(tracks[...,1]-(h-1)).clamp_min(0))/distance_scale
    use=torch.tensor(valid,device=tracks.device)[:,None] & visible.bool()
    return (value+outside)[use].mean() if bool(use.any()) else tracks.sum()*0
