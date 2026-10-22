"""Label quality probes; no student, optimizer, or adoption decision."""
import numpy as np


def review(outputs, queries, height, width, primary, threshold=2.):
    """Numpy mirror of collaboration v1; runtime checks against executed loss hook."""
    p, v = outputs[primary][:2]
    after = np.arange(p.shape[1])[None, :, None] > queries[:, None, :, 0]
    def supported(item):
        xy, vis = item[:2]
        return (after & np.isfinite(xy).all(-1) & np.isfinite(vis) & (vis > .9) &
                (xy[..., 0] >= 0) & (xy[..., 0] <= width - 1) &
                (xy[..., 1] >= 0) & (xy[..., 1] <= height - 1))
    active = v > .9  # Diagnostic valids are all one, as in the input dataset.
    valid = supported(outputs[primary]) & active
    other = [outputs[i] for i in range(3) if i != primary]
    ok = [supported(item) & valid for item in other]
    a, b = [np.where(mask[..., None], item[0], p) for item, mask in zip(other, ok)]
    joint = ok[0] & ok[1]
    da, db, dab = [np.linalg.norm(x, axis=-1) for x in (a-p, b-p, a-b)]
    conflict = joint & (dab <= threshold) & (da > threshold) & (db > threshold)
    agreement = joint & (dab <= threshold) & (da <= threshold) & (db <= threshold)
    mean = (a + b) * .5
    correction = .25 * (mean - p)
    correction *= np.minimum(1., .5 / np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-12))
    fused = np.where(agreement[..., None], p + correction, p)
    weights = np.where(conflict, .25, 1.).astype(np.float32)
    return dict(primary=p, reviewer_a=a, reviewer_b=b, reviewer_mean=mean, fused=fused,
                active=active, eligible=joint, conflict=conflict, agreement=agreement,
                changed=np.linalg.norm(fused-p, axis=-1) > 0, weights=weights)


def matrices(kind, frames, height, width):
    """Forward source-to-target transforms in pixel coordinates, frame zero identity."""
    if kind not in ('static', 'translation', 'rotation_zoom'):
        raise ValueError('Unknown motion')
    result = []
    center = np.array([(width-1)/2, (height-1)/2])
    for t in range(frames):
        f = t / max(1, frames-1)
        angle = np.deg2rad(12*f) if kind == 'rotation_zoom' else 0.
        scale = 1 + .05*f if kind == 'rotation_zoom' else 1.
        shift = np.array([12*f, -8*f]) if kind != 'static' else np.zeros(2)
        linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        mat = np.eye(3)
        mat[:2, :2] = linear
        mat[:2, 2] = center + shift - linear @ center
        result.append(mat)
    return np.stack(result).astype(np.float32)


def transform_points(xy, transforms):
    xy1 = np.concatenate([xy, np.ones((*xy.shape[:-1], 1), dtype=xy.dtype)], -1)
    return np.einsum('tij,nj->tni', transforms, xy1)[..., :2]


def known_errors(predictions, truth, height, width):
    valid = (np.isfinite(truth).all(-1) & (truth[..., 0] >= 1) & (truth[..., 0] <= width-2) &
             (truth[..., 1] >= 1) & (truth[..., 1] <= height-2))
    errors = {k: np.where(valid, np.linalg.norm(v-truth, axis=-1), np.nan) for k, v in predictions.items()}
    return errors, valid


def boundary_errors(predictions, masks):
    """Unsigned distance to annotated boundary: NOT point correspondence accuracy."""
    from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
    shape = next(iter(predictions.values())).shape[:-1]
    result = {k: np.full(shape, np.nan) for k in predictions}
    h, w = masks.shape[-2:]
    for t, mask in enumerate(masks):
        if not mask.any() or mask.all():
            continue
        boundary = mask & ~binary_erosion(mask)
        field = distance_transform_edt(~boundary)
        for name, points in predictions.items():
            xy = points[0, t]
            valid = (np.isfinite(xy).all(-1) & (xy[:, 0] >= 0) & (xy[:, 0] <= w-1) &
                     (xy[:, 1] >= 0) & (xy[:, 1] <= h-1))
            result[name][0, t, valid] = map_coordinates(field, [xy[valid, 1], xy[valid, 0]], order=1, mode='nearest')
    return result


def summarize(mask, errors):
    """Every comparison uses the same finite population. Empty stays missing."""
    common = mask.copy()
    for values in errors.values():
        common &= np.isfinite(values)
    n = int(common.sum())
    row = {'Selected': int(mask.sum()), 'Scored': n, 'Unscored': int(mask.sum()) - n}
    def mean(value):
        return float(value[common].mean()) if n else None
    for name, values in errors.items():
        row[name + 'Error'] = mean(values)
    primary, reviewer, fused = [errors[k] for k in ('Primary', 'ReviewerMean', 'Fused')]
    row.update(ReviewerMinusPrimary=mean(reviewer-primary), FusedMinusPrimary=mean(fused-primary),
               ReviewerBetterFraction=mean(reviewer < primary-1e-6),
               FusedBetterFraction=mean(fused < primary-1e-6),
               FusedWorseFraction=mean(fused > primary+1e-6),
               PrimaryErrorAbove2Fraction=mean(primary > 2),
               BothReviewersBetterFraction=mean((errors['ReviewerA'] < primary-1e-6) & (errors['ReviewerB'] < primary-1e-6)),
               AllThreeErrorAbove2Fraction=mean((primary > 2) & (errors['ReviewerA'] > 2) & (errors['ReviewerB'] > 2)))
    return row


def aggregate(rows):
    """Case-macro headlines and point-frame micro values, never pool modalities."""
    groups = {}
    ids = ('Mode', 'Motion', 'QueryGroup', 'PrimarySlot', 'Threshold', 'Cohort')
    for r in rows:
        key = tuple(r[k] for k in ids)
        groups.setdefault(key, []).append(r)
    answer = []
    for key, values in sorted(groups.items()):
        if len({r['Case'] for r in values}) != len(values):
            raise ValueError('Duplicate case in aggregate')
        out = dict(zip(ids, key))
        good = [r for r in values if r['Scored'] > 0]
        out.update(CasesAttempted=len(values), CasesScored=len(good),
                   Selected=sum(r['Selected'] for r in values), Scored=sum(r['Scored'] for r in values),
                   Unscored=sum(r['Unscored'] for r in values))
        for name in values[0]:
            if name in ids or name in ('Case', 'Selected', 'Scored', 'Unscored'):
                continue
            out['Macro_'+name] = float(np.mean([r[name] for r in good])) if good else None
            out['Micro_'+name] = sum(r[name]*r['Scored'] for r in good)/out['Scored'] if good else None
        answer.append(out)
    return answer


def contrasts(rows):
    """Small case-paired mechanism table; no significance or automatic adoption."""
    groups = {}
    ids = ('Mode', 'Motion', 'PrimarySlot', 'Threshold')
    for r in rows:
        if r['QueryGroup'] != 'all':
            continue
        group = groups.setdefault(tuple(r[k] for k in ids), {})
        case = group.setdefault(r['Case'], {})
        if r['Cohort'] in case:
            raise ValueError('Duplicate cohort')
        case[r['Cohort']] = r
    result = []
    def avg(xs):
        return float(np.mean(xs)) if xs else None
    for key, cases in sorted(groups.items()):
        if any(set(c) != {'eligible', 'conflict', 'nonconflict', 'shuffled', 'agreement', 'fused_changed'} for c in cases.values()):
            raise ValueError('Incomplete cohorts')
        conflict = [c['conflict'] for c in cases.values() if c['conflict']['Scored']]
        paired = [c for c in cases.values() if c['conflict']['Scored'] and c['shuffled']['Scored']]
        fused = [c['fused_changed'] for c in cases.values() if c['fused_changed']['Scored']]
        agree = [c['agreement'] for c in cases.values() if c['agreement']['Scored']]
        row = dict(zip(ids, key))
        row.update(Cases=len(cases), ConflictCases=len(conflict), PairedConflictShuffleCases=len(paired),
            ConflictScored=sum(c['conflict']['Scored'] for c in cases.values()),
            ConflictUnscored=sum(c['conflict']['Unscored'] for c in cases.values()),
            ShuffledScored=sum(c['shuffled']['Scored'] for c in cases.values()),
            ShuffledUnscored=sum(c['shuffled']['Unscored'] for c in cases.values()),
            MacroConflictMinusShuffledPrimaryError=avg([c['conflict']['PrimaryError']-c['shuffled']['PrimaryError'] for c in paired]),
            MacroReviewMinusPrimaryOnConflict=avg([c['ReviewerMinusPrimary'] for c in conflict]),
            MacroBothReviewersBetterOnConflict=avg([c['BothReviewersBetterFraction'] for c in conflict]),
            MacroAllThreeBadOnAgreement=avg([c['AllThreeErrorAbove2Fraction'] for c in agree]),
            FusedCases=len(fused), FusedScored=sum(c['Scored'] for c in fused),
            MacroFusedMinusPrimaryOnChanged=avg([c['FusedMinusPrimary'] for c in fused]),
            FusedImprovedCases=sum(c['FusedMinusPrimary'] < -1e-6 for c in fused),
            FusedWorsenedCases=sum(c['FusedMinusPrimary'] > 1e-6 for c in fused),
            FusedTiedCases=sum(abs(c['FusedMinusPrimary']) <= 1e-6 for c in fused))
        result.append(row)
    return result
