"""Training-only GRL; never installed into the inference state dict."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from protocol import ARMS, PREFIX, ENCODER_PREFIXES, RULE


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def reversal(x, strength):
    if not math.isfinite(strength) or strength < 0: raise ValueError('Invalid GRL strength')
    # Detached zero control must not add even zero gradients to the student graph.
    return x.detach() if strength == 0 else Reverse.apply(x, strength)


def schedule(step, maximum):
    if step < 0: raise ValueError('Negative step')
    p = min(step / RULE['warmup_steps'], 1.)
    return maximum * (2/(1+math.exp(-5*p))-1)/(2/(1+math.exp(-5))-1)


class DomainHead(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.LayerNorm(width), nn.Linear(width,128), nn.GELU(), nn.Linear(128,1))


def configure(model, arm):
    if arm not in ARMS: raise ValueError('Unknown arm')
    model.eval()
    model.updateformer.feature_sidecar.train()
    partial = arm.startswith('F')
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(PREFIX) or (partial and name.startswith(ENCODER_PREFIXES)))
    # Keep normalization/eval behavior identical in all arms. Frozen Transformer weights
    # still participate in autograd when encoder outputs require gradients.
    return partial


class Tap:
    def __init__(self, model, kind):
        self.kind = kind; self.values = []; self.frames = None
        target = model.fnet if kind == 'fnet' else model.updateformer.feature_sidecar.blocks[-1]
        self.handle = target.register_forward_hook(self.capture)
    def reset(self, frames):
        if frames < 2: raise ValueError('Need two native frames')
        self.frames = frames; self.values.clear()
    def capture(self, module, inputs, output):
        if self.frames is None: raise ValueError('Tap must be reset before forward')
        if self.kind == 'fnet':
            if output.ndim != 4: raise ValueError('Expected T,C,H,W fnet features')
            self.values.append(output.mean(dim=(-2,-1)))
        else:
            if output.ndim != 3 or output.shape[1] != self.frames:
                raise ValueError('Sidecar must expose actual native sequence, batch=1')
            # Last refinement only; replacing pooled value releases the extra earlier tap graph.
            self.values[:] = [output[:,1:].mean(dim=(0,1), keepdim=False)[None]]
    def embedding(self):
        if not self.values: raise ValueError('Domain hook was bypassed')
        if self.kind == 'sidecar': return self.values[0]
        frames = torch.cat(self.values)
        if len(frames) != self.frames: raise ValueError('Fnet chunks do not match native frames')
        return frames[1:].mean(0,keepdim=True)
    def close(self):
        self.handle.remove(); self.values.clear()


def domain_loss(head, embedding, domain, strength):
    if domain not in ('trackrad','acdc') or embedding.shape[0] != 1:
        raise ValueError('Domain loss expects one native clip')
    logits = head(reversal(embedding,strength)).flatten()
    target = torch.full_like(logits, float(domain == 'acdc'))
    return F.binary_cross_entropy_with_logits(logits,target), logits.detach().sigmoid()


def gradient_norm(loss, params, retain_graph=True):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    values = [g.detach().float().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.
