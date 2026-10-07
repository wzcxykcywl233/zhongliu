"""Frozen Transformer + per-point feature residual sidecar.

Mamba-1-style PyTorch reference selective scan, NOT official mamba-ssm CUDA
kernels. States are independent across points, clips and refinement iterations.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ('mlp', 'conv', 'mamba_short', 'mamba')


class SelectiveScan(nn.Module):
    def __init__(self, width=256, state=16):
        super().__init__()
        inner, rank = width * 2, math.ceil(width / 16)
        self.inner, self.state = inner, state
        self.in_proj = nn.Linear(width, inner * 2, bias=False)
        self.conv = nn.Conv1d(inner, inner, 4, groups=inner, padding=3)
        self.x_proj = nn.Linear(inner, rank + state * 2, bias=False)
        self.dt_proj = nn.Linear(rank, inner)
        self.a_log = nn.Parameter(torch.arange(1, state + 1).float().log().repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, width, bias=False)
        nn.init.constant_(self.dt_proj.bias, -4.)

    def scan(self, x):
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        dt, b, c = torch.split(self.x_proj(u), [self.dt_proj.in_features, self.state, self.state], -1)
        dt = F.softplus(self.dt_proj(dt)).float()
        a = -self.a_log.float().exp()
        h = torch.zeros((x.shape[0], self.inner, self.state), device=x.device, dtype=torch.float32)
        outputs = []
        for t in range(x.shape[1]):
            h = torch.exp(dt[:, t, :, None] * a) * h + dt[:, t, :, None] * b[:, t, None].float() * u[:, t, :, None].float()
            outputs.append((h * c[:, t, None].float()).sum(-1) + self.skip.float() * u[:, t].float())
        y = torch.stack(outputs, 1).to(x.dtype) * F.silu(z)
        return self.out_proj(y)

    def forward(self, x):
        return .5 * (self.scan(x) + self.scan(x.flip(1)).flip(1))


class SideBlock(nn.Module):
    def __init__(self, kind, width=256):
        super().__init__()
        self.norm = nn.Identity() if kind == 'mlp' else nn.LayerNorm(width)
        self.kind = kind
        if kind == 'mamba':
            self.mix = SelectiveScan(width)
            inner = 1024
        else:
            self.mix = nn.Conv1d(width, width, 5, padding=2, groups=width) if kind == 'conv' else nn.Identity()
            inner = 1792
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, inner), nn.GELU(), nn.Linear(inner, width))

    def forward(self, x):
        if self.kind != 'mlp':
            y = self.norm(x)
            y = self.mix(y.transpose(1, 2)).transpose(1, 2) if self.kind == 'conv' else self.mix(y)
            x = x + y
        return x + self.ffn(self.ffn_norm(x))


class FeatureSidecar(nn.Module):
    def __init__(self, kind, input_dim=1110, hidden_dim=384, width=256, depth=3):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('unknown feature sidecar')
        self.kind = kind
        self.input_norm = nn.LayerNorm(input_dim + hidden_dim)
        self.input_proj = nn.Linear(input_dim + hidden_dim, width)
        self.blocks = nn.ModuleList(SideBlock('mamba' if kind == 'mamba_short' else kind, width) for _ in range(depth))
        self.gate = nn.Linear(width, 1)
        self.output = nn.Linear(width, hidden_dim, bias=False)
        # Zero ONLY the last projection; the sigmoid gate stays nonzero.
        nn.init.zeros_(self.output.weight)
        self.last_rms = 0.

    def forward(self, hidden, rich):
        return hidden.detach() + self.residual(hidden, rich)

    def residual(self, hidden, rich):
        batch, points, frames, _ = hidden.shape
        source = torch.cat((rich.detach(), hidden.detach()), -1)
        x = self.input_proj(self.input_norm(source)).reshape(batch * points, frames, -1)
        if self.kind == 'mamba_short':
            x = torch.cat([self.run_blocks(x[:, start:start + 10]) for start in range(0, frames, 10)], 1)
        else:
            x = self.run_blocks(x)
        residual = self.output(x) * torch.sigmoid(self.gate(x)) * .1
        if not self.training:
            self.last_rms = float(residual.detach().square().mean().sqrt())
            d = getattr(self, 'diagnostics', None)
            if d is not None:
                d['sidecar_calls'] = d.get('sidecar_calls', 0) + 1
                d['sidecar_sequence_frames_max'] = max(d.get('sidecar_sequence_frames_max', 0), frames)
                d['sidecar_residual_rms_max'] = max(d.get('sidecar_residual_rms_max', 0.), self.last_rms)
        return residual.reshape_as(hidden)

    def run_blocks(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if '.feature_sidecar.' in key:
            continue
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def attach_sidecar(model, kind):
    if kind not in KINDS or hasattr(model.updateformer, 'feature_sidecar'):
        raise ValueError('invalid or duplicate sidecar attachment')
    if not model.updateformer.linear_layer_for_vis_conf:
        raise ValueError('feature sidecar requires separate coordinate and V/C heads')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.updateformer.feature_sidecar = FeatureSidecar(kind, model.updateformer.input_transform.in_features,
                                                       model.updateformer.flow_head.in_features)
    model.feature_sidecar_kind = kind
    model.frozen_pretrained = True
    return {'kind': kind, 'base_tensor_sha256': tensor_hash(model.state_dict()),
            'initial_sidecar_sha256': tensor_hash(model.updateformer.feature_sidecar.state_dict()),
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'zero_projection_verified': bool((model.updateformer.feature_sidecar.output.weight == 0).all()),
            'backend': 'pytorch-reference-selective-ssm' if kind.startswith('mamba') else kind}


def attach_sidecar_verified(model, kind):
    """Small real image/correlation/two-iteration proof before any optimization."""
    threads, mode = torch.get_num_threads(), model.training
    torch.set_num_threads(1)
    model.eval()
    try:
        generator = torch.Generator().manual_seed(20261008)
        video = torch.rand(1, 3, 3, 64, 64, generator=generator) * 255
        queries = torch.tensor([[[0.,16.,16.], [1.,32.,32.], [0.,48.,48.]]])
        with torch.no_grad():
            original = tuple(x.clone() for x in model(video=video, queries=queries, iters=2)[:3])
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            before = tuple(x.clone() for x in model(video=video, queries=queries, iters=2)[:3])
            report = attach_sidecar(model, kind)
            after = model(video=video, queries=queries, iters=2)[:3]
        freeze_differences = [float((a-b).abs().max()) for a,b in zip(original,before)]
        if any(d > limit for d,limit in zip(freeze_differences,(1e-4,1e-6,1e-6))):
            raise ValueError('unexpected numerical change from freezing base: ' + str(freeze_differences))
        report['freeze_transition_max_abs_difference'] = freeze_differences
        differences = [float((a-b).abs().max()) for a,b in zip(before,after)]
        if any(differences):
            raise ValueError('sidecar zero initialization changed model outputs: ' + str(differences))
        report['initialization_outputs_exact'] = True
        report['initialization_max_abs_difference'] = differences
        return report
    finally:
        model.train(mode)
        torch.set_num_threads(threads)
