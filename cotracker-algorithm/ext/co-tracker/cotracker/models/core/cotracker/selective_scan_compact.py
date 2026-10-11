"""Sequential selective scan with sparse states and first-order analytic backward.

No [points,time,channels,state] alpha/beta tensor or per-time autograd graph.
Save a state BEFORE every eight frames; recompute that short span in backward.
The recurrence, state lifetime and forward arithmetic order match reference.
This is still PyTorch, not an official fused Mamba CUDA kernel. No double grad.
"""
import torch
from torch.autograd.function import once_differentiable


def validate(u, delta, a, b, c, skip):
    if u.ndim!=3 or delta.shape!=u.shape or a.ndim!=2:
        raise ValueError('Expected u/delta [points,time,channels], a [channels,state]')
    p,t,h=u.shape
    if not p or not t or not h or a.shape[1]<1 or a.shape[0]!=h or b.shape!=(p,t,a.shape[1]) or c.shape!=b.shape or skip.shape!=(h,):
        raise ValueError('Selective scan shape mismatch')
    if any(x.device!=u.device or x.dtype!=u.dtype for x in (delta,a,b,c,skip)):
        raise ValueError('Selective scan requires one device and dtype')
    if u.dtype not in (torch.float32,torch.float64): raise ValueError('FP32/FP64 scan only')


def advance(previous, u, delta, a, b):
    # Keep reference left-to-right multiplication, no division by decay and
    # no reset/detach between sparse checkpoint spans.
    decay=torch.exp(delta[...,None]*a)
    return decay*previous + delta[...,None]*b[:,None,:]*u[...,None]


class CompactSelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, a, b, c, skip):
        validate(u,delta,a,b,c,skip)
        p,t,h=u.shape
        state=torch.zeros((p,h,a.shape[1]),device=u.device,dtype=u.dtype)
        anchors=[]; values=[]
        for index in range(t):
            if index%8==0: anchors.append(state)
            state=advance(state,u[:,index],delta[:,index],a,b[:,index])
            values.append((state*c[:,index,None,:]).sum(-1)+skip*u[:,index])
        ctx.save_for_backward(u,delta,a,b,c,skip,torch.stack(anchors))
        return torch.stack(values,1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        u,delta,a,b,c,skip,anchors=ctx.saved_tensors
        p,t,h=u.shape
        gu=torch.zeros_like(u); gd=torch.zeros_like(delta); ga=torch.zeros_like(a)
        gb=torch.zeros_like(b); gc=torch.zeros_like(c); gs=torch.zeros_like(skip)
        carry=torch.zeros((p,h,a.shape[1]),device=u.device,dtype=u.dtype)
        for start in reversed(range(0,t,8)):
            end=min(start+8,t)
            states=[anchors[start//8]]
            for index in range(start,end):
                states.append(advance(states[-1],u[:,index],delta[:,index],a,b[:,index]))
            for index in reversed(range(start,end)):
                previous,current=states[index-start:index-start+2]
                d=delta[:,index,:,None]; inp=u[:,index,:,None]
                bt=b[:,index,None,:]; ct=c[:,index,None,:]
                decay=torch.exp(d*a)
                gy=grad_y[:,index]
                adjoint=carry+gy[...,None]*ct
                gc[:,index]=(gy[...,None]*current).sum(1)
                gb[:,index]=(adjoint*d*inp).sum(1)
                gu[:,index]=(adjoint*d*bt).sum(-1)+gy*skip
                gd[:,index]=(adjoint*(a*decay*previous+bt*inp)).sum(-1)
                ga.add_((adjoint*d*decay*previous).sum(0))
                gs.add_((gy*u[:,index]).sum(0))
                carry=adjoint*decay
            del states
        gradients=(gu,gd,ga,gb,gc,gs)
        return tuple(g if needed else None for g,needed in zip(gradients,ctx.needs_input_grad))


def compact_selective_scan(u, delta, a, b, c, skip):
    return CompactSelectiveScan.apply(u,delta,a,b,c,skip)
