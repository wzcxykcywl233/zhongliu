"""Observation-only per-frame, same-offset feature backcheck (no labels)."""
import math
import torch
import torch.nn.functional as F


def sample(feature, xy, stride):
    h, w = feature.shape[-2:]
    coordinates = xy / stride
    grid = torch.stack((2 * coordinates[..., 0] / (w - 1) - 1,
                        2 * coordinates[..., 1] / (h - 1) - 1), -1)
    values = F.grid_sample(feature, grid[:, None], align_corners=True,
                           mode='bilinear', padding_mode='zeros')[:, :, 0].transpose(1, 2)
    return F.normalize(values.float(), dim=-1, eps=1e-8)


def inside(xy, feature, stride):
    h, w = feature.shape[-2:]
    return ((xy[..., 0] >= 0) & (xy[..., 0] <= (w-1)*stride)
            & (xy[..., 1] >= 0) & (xy[..., 1] <= (h-1)*stride))


def score_frames(features, queries, final, iterations, start, stride=4):
    """Full point set, level-zero features. Queries refer to this segment's t=0.

    Use updates 1..I-1 only, never initial guess or last iteration as evidence.
    History and fixed probes share exactly the same accepted sample mask.
    """
    if final.shape[0] != 1 or features.shape[:2] != final.shape[:2]:
        raise ValueError('backcheck expects one matching video/trajectory batch')
    if len(iterations) < 2 or any(x.shape != final.shape for x in iterations):
        raise ValueError('backcheck needs >=2 same-shaped iteration trajectories')
    if not all(bool(torch.isfinite(x).all()) for x in (features, queries, final, *iterations)):
        raise ValueError('nonfinite backcheck input')
    q = queries[..., 1:3].to(features.device)
    reference = features[:, 0]
    query_feature = sample(reference, q, stride)
    rows = []
    for t in range(1, final.shape[1]):
        p = final[:, t]
        target = features[:, t]
        center = (query_feature * sample(target, p, stride)).sum(-1).clamp(-1,1)
        center_valid = inside(q, reference, stride) & inside(p, target, stride)
        history_sum = torch.zeros_like(center)
        fixed_sum = torch.zeros_like(center)
        counts = torch.zeros_like(center)
        accepted_offsets = []
        radii = []
        # Fixed probes: radius 4 model pixels, equally spaced around a circle.
        n = len(iterations) - 1
        for index, trajectory in enumerate(iterations[:-1]):
            delta = trajectory[:, t] - p
            radius = torch.linalg.vector_norm(delta, dim=-1)
            angle = 2 * math.pi * index / n
            fixed = p.new_tensor([4 * math.cos(angle), 4 * math.sin(angle)])
            valid = center_valid & (radius >= .5) & (radius <= 16.)
            for old_delta, old_valid in accepted_offsets:
                valid &= (~old_valid) | (torch.linalg.vector_norm(delta-old_delta, dim=-1) >= .5)
            valid &= inside(q+delta, reference, stride) & inside(p+delta, target, stride)
            valid &= inside(q+fixed, reference, stride) & inside(p+fixed, target, stride)
            hs = (sample(reference, q+delta, stride) * sample(target, p+delta, stride)).sum(-1).clamp(-1,1)
            fs = (sample(reference, q+fixed, stride) * sample(target, p+fixed, stride)).sum(-1).clamp(-1,1)
            history_sum += torch.where(valid, hs, 0.)
            fixed_sum += torch.where(valid, fs, 0.)
            counts += valid.float()
            accepted_offsets.append((delta, valid.clone()))
            radii.append(radius[valid])
        supported = counts > 0
        def mean_or_none(values):
            return float(values.mean().item()) if values.numel() else None
        history = history_sum / counts.clamp_min(1)
        fixed_score = fixed_sum / counts.clamp_min(1)
        # SAME supported point population for all three competing frame scores.
        row = {
            'frame': start+t, 'query_frame': start, 'points': p.shape[1],
            'center_valid_points': int(center_valid.sum()), 'supported_points': int(supported.sum()),
            'accepted_probes': int(counts.sum()), 'possible_probes': p.shape[1]*n,
            'mean_radius': mean_or_none(torch.cat(radii)),
            'center_all': mean_or_none(center[center_valid]),
            'center': mean_or_none(center[supported]),
            'history': mean_or_none(history[supported]),
            'fixed': mean_or_none(fixed_score[supported]),
            'center_history': mean_or_none(((center+history)/2)[supported]),
            'center_fixed': mean_or_none(((center+fixed_score)/2)[supported]),
        }
        rows.append(row)
    return rows


def score_fourway_frames(features, queries, final, iterations, start, stride=4):
    """Observation only. Four scores share points with FOUR valid distinct probes.

    At two iterations there is one historical update before the final update.
    Reflect its offset about x/y axes; do not rotate or search for a new center.
    Axis-degenerate and out-of-bounds groups are excluded, not zero-filled.
    """
    if final.shape[0] != 1 or features.shape[:2] != final.shape[:2]:
        raise ValueError('four-way backcheck expects one aligned video batch')
    if len(iterations) != 2 or any(x.shape != final.shape for x in iterations):
        raise ValueError('four-way backcheck requires exactly two iterations')
    if not all(bool(torch.isfinite(x).all()) for x in (features, queries, final, *iterations)):
        raise ValueError('nonfinite four-way input')
    # Explicit device/dtype alignment prevents CPU/GPU mixed grid sampling.
    final = final.to(device=features.device, dtype=torch.float32)
    previous = iterations[0].to(device=features.device, dtype=torch.float32)
    q = queries[..., 1:3].to(device=features.device, dtype=torch.float32)
    reference = features[:, 0].float()
    signs = final.new_tensor([[1,1],[-1,1],[-1,-1],[1,-1]])
    fixed = signs * (4 / math.sqrt(2))  # radius 4, diagonal four-way control
    rows = []
    def mean_or_none(values):
        return float(values.mean().item()) if values.numel() else None
    for t in range(1, final.shape[1]):
        p, target = final[:,t], features[:,t].float()
        delta = previous[:,t] - p
        radius = torch.linalg.vector_norm(delta, dim=-1)
        offsets = delta[...,None,:] * signs
        center_valid = inside(q,reference,stride) & inside(p,target,stride)
        valid_radius = (radius >= .5) & (radius <= 16)
        # Pairwise distance >= .5 guarantees four distinct offsets. In
        # particular dx=0 or dy=0 would duplicate two pairs and is rejected.
        distinct = torch.ones_like(center_valid)
        bounds = torch.ones_like(center_valid)
        for j in range(4):
            for k in range(j):
                distinct &= torch.linalg.vector_norm(offsets[...,j,:]-offsets[...,k,:],dim=-1) >= .5
            for offset in (offsets[...,j,:],fixed[j]):
                bounds &= inside(q+offset,reference,stride) & inside(p+offset,target,stride)
        supported = center_valid & valid_radius & distinct & bounds
        def similarity(offset):
            return (sample(reference,q+offset,stride)*sample(target,p+offset,stride)).sum(-1).clamp(-1,1)
        center = similarity(p.new_zeros(2))
        hs = torch.stack([similarity(offsets[...,j,:]) for j in range(4)],dim=-1)
        fs = torch.stack([similarity(fixed[j]) for j in range(4)],dim=-1)
        history4, fixed4 = hs.mean(-1), fs.mean(-1)
        row = dict(frame=start+t, query_frame=start, points=p.shape[1],
            center_valid_points=int(center_valid.sum()), supported_points=int(supported.sum()),
            radius_rejected_points=int((center_valid & ~valid_radius).sum()),
            degenerate_rejected_points=int((center_valid & valid_radius & ~distinct).sum()),
            bounds_rejected_points=int((center_valid & valid_radius & distinct & ~bounds).sum()),
            accepted_probes=4*int(supported.sum()), possible_probes=4*p.shape[1],
            mean_radius=mean_or_none(radius[supported]),
            center_all=mean_or_none(center[center_valid]), center=mean_or_none(center[supported]),
            history_single=mean_or_none(hs[...,0][supported]),
            history_four=mean_or_none(history4[supported]), fixed_four=mean_or_none(fixed4[supported]),
            center_history=mean_or_none(((center+hs[...,0])/2)[supported]),
            center_history_four=mean_or_none(((center+history4)/2)[supported]),
            center_fixed_four=mean_or_none(((center+fixed4)/2)[supported]),
            history_direction_range=mean_or_none((hs.max(-1).values-hs.min(-1).values)[supported]))
        for j in range(4):
            row[f'history_direction_{j}'] = mean_or_none(hs[...,j][supported])
            row[f'fixed_direction_{j}'] = mean_or_none(fs[...,j][supported])
        rows.append(row)
    return rows
