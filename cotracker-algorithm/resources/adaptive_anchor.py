"""Per-point rollback decisions. Scores are evidence, not calibrated probabilities."""
import torch


def last_stable_offsets(visibility, confidence, query_offsets, absolute=.5,
                        relative=.15, patience=2, window=3):
    """First sustained drop -> the frame BEFORE its onset, independently per point.

    Baselines use only preceding stable frames; the supplied query is initially
    trusted. Never use a future recovery to promote an already dropped anchor.
    """
    if visibility.shape != confidence.shape or visibility.ndim != 3:
        raise ValueError('scores must be matching B,T,N tensors')
    if query_offsets.shape != visibility[:, 0].shape:
        raise ValueError('query offsets must be B,N')
    if not 0 <= absolute <= 1 or not 0 < relative < 1 or patience < 1 or window < 1:
        raise ValueError('invalid rollback thresholds')
    if not bool(torch.isfinite(visibility).all() & torch.isfinite(confidence).all()):
        raise ValueError('nonfinite scores')
    if bool((query_offsets < 0).any() | (query_offsets >= visibility.shape[1]).any()):
        raise ValueError('query outside clip')
    end = visibility.shape[1] - 1
    chosen = torch.full_like(query_offsets, end)
    triggered = torch.zeros_like(query_offsets, dtype=torch.bool)
    # This loop is over time only, not N individual model invocations.
    baseline_v = torch.zeros_like(visibility[:, 0])
    baseline_c = torch.zeros_like(baseline_v)
    streak = torch.zeros_like(query_offsets)
    onset = query_offsets.clone()
    for t in range(end + 1):
        at_query = query_offsets == t
        baseline_v = torch.where(at_query, visibility[:, t], baseline_v)
        baseline_c = torch.where(at_query, confidence[:, t], baseline_c)
        active = (t > query_offsets) & ~triggered
        bad = active & ((visibility[:, t] < absolute) | (confidence[:, t] < absolute)
                        | (visibility[:, t] < baseline_v - relative)
                        | (confidence[:, t] < baseline_c - relative))
        onset = torch.where(bad & (streak == 0), torch.full_like(onset, t), onset)
        streak = torch.where(bad, streak + 1, torch.zeros_like(streak))
        drop = active & (streak >= patience)
        chosen = torch.where(drop, torch.maximum(query_offsets, onset - 1), chosen)
        triggered |= drop
        healthy = active & ~bad
        baseline_v = torch.where(healthy, baseline_v + (visibility[:, t] - baseline_v) / window, baseline_v)
        baseline_c = torch.where(healthy, baseline_c + (confidence[:, t] - baseline_c) / window, baseline_c)
    # A pending (not yet sustained) drop is not a safe new endpoint either.
    pending = (~triggered) & (streak > 0)
    chosen = torch.where(pending, torch.maximum(query_offsets, onset - 1), chosen)
    return chosen, triggered, pending
