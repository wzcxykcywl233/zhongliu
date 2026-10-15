"""Arbitrary-time, single-factor offline teacher interventions (no re-anchoring)."""
import hashlib

import torch

POLICIES = ('old', 'scaled', 'grid0', 'iter6')
SETTINGS = {
    'old': ('baseline_offline.pth', 100, 4),
    'scaled': ('scaled_offline.pth', 100, 4),
    'grid0': ('scaled_offline.pth', 0, 4),
    'iter6': ('scaled_offline.pth', 0, 6),
}


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def offline_teacher_forward(model, video, queries, support, policy):
    """Always consume the SAME support draw upstream, even when it is unused.

    Query t/x/y and order are never modified. Only the offline slot changes;
    checkpoint selection is enforced by the launcher and the trainer.
    """
    _, count, iterations = SETTINGS[policy]
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('offline teacher must be frozen/eval')
    if queries.shape[0] != 1 or queries.shape[-1] != 3 or support.shape != (1, 100, 3):
        raise ValueError('expected batch=1 t/x/y queries and a matched 100-point draw')
    if not torch.isfinite(queries).all() or not torch.isfinite(support).all():
        raise ValueError('nonfinite teacher queries')
    times = queries[..., 0]
    if bool(((times < 0) | (times >= video.shape[1]) | (times != times.round())).any()):
        raise ValueError('invalid query times; never reset them to zero')
    combined = torch.cat((queries, support), 1) if count else queries
    with torch.no_grad():
        result = model(video, combined, iters=iterations)
    n = queries.shape[1]
    return tuple(value[:, :, :n] for value in result[:3])


def label_audit(video, queries, tracks, visibility, confidence, *, policy,
                teacher_type, support_draw_sha256, seconds):
    """Raw labels before V>.9 binarization; diagnostics, not calibrated quality."""
    b, t, n = video.shape[0], video.shape[1], queries.shape[1]
    if tracks.shape != (b, t, n, 2) or any(x.shape != (b, t, n) for x in (visibility, confidence)):
        raise ValueError('pseudo-label shape mismatch')
    if not all(bool(torch.isfinite(x).all()) for x in (tracks, visibility, confidence)):
        raise ValueError('nonfinite pseudo labels')
    if any(bool(((x < 0) | (x > 1)).any()) for x in (visibility, confidence)):
        raise ValueError('teacher V/C outside [0,1]')
    p, v, c = (x.detach() for x in (tracks, visibility, confidence))
    outside = (p[..., 0] < 0) | (p[..., 0] >= video.shape[-1]) | (p[..., 1] < 0) | (p[..., 1] >= video.shape[-2])
    return {
        'policy': policy, 'teacher_type': teacher_type,
        'support_draw_sha256': support_draw_sha256,
        'tracks_sha256': tensor_sha(p), 'visibility_sha256': tensor_sha(v),
        'confidence_sha256': tensor_sha(c), 'teacher_seconds': seconds,
        'visible_gt09_fraction': float((v > .9).float().mean()),
        'confidence_mean': float(c.mean()), 'out_of_bounds_fraction': float(outside.float().mean()),
        'query_time_counts': torch.bincount(queries[..., 0].long().flatten(), minlength=t).cpu().tolist(),
        'support_used': SETTINGS[policy][1] if teacher_type == 'offline_cotracker_three' else None,
        'iterations': SETTINGS[policy][2] if teacher_type == 'offline_cotracker_three' else None,
    }
