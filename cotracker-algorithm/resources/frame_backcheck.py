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
