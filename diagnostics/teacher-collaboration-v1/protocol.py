"""Teacher review changes only visible coordinate supervision, not the student."""
from contextlib import contextmanager
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from shared import base_protocol as B

BASE_SEED, SEEDS, PROFILE, RUNTIME, RULE = B.BASE_SEED, B.SEEDS, B.PROFILE, B.RUNTIME, B.RULE
FIELDS, INPUT_KEYS, LABEL_KEYS = B.FIELDS, B.INPUT_KEYS, B.LABEL_KEYS
STEPS = {'smoke': 10, 'pilot': 250, 'full': 1000}
read, sha, digest = B.read, B.sha, B.digest
means, ranking, best, selection_evidence = B.means, B.ranking, B.best, B.selection_evidence
TEACHER_TYPES = ('online', 'online_cotracker_three', 'offline_cotracker_three')
ARMS = ('control', 'tutor010', 'gate2', 'gate4', 'shuffle2', 'fuse2')
NOOPS = ('control', 'shadow', 'same_teacher')
COLLABORATION_RULE = {
    'teacher_pool': B.baseline(), 'arms': list(ARMS), 'smoke_only': ['shadow', 'same_teacher'],
    'review_visibility': .9, 'outlier_weight': .25, 'fusion_fraction': .25,
    'maximum_correction_pixels': .5, 'tutor_total_weight': .1,
    'thresholds_training_pixels': [2., 4.], 'student_forward_passes': 1,
    'coordinate_loss': 'visible Huber only; original V/C and invisible losses unchanged',
    'normalization': 'weighted coordinate mean; fixed total loss coefficient, not fixed gradient norm',
    'conflict': 'two valid visible reviewers agree; primary is farther than threshold from BOTH',
    'fusion': 'all three pair distances <= threshold; query frame never changed',
    'invalid': 'nonfinite primary aborts; invalid/occluded reviewer abstains; no new coverage',
    'selection': 'pilot diagnostic only; all 6 fixed arms confirm with 3 seeds; validation selects; test reports all',
}


def matrix():
    return {a: {'teachers': B.baseline(), 'collaboration': a} for a in (*ARMS, 'shadow', 'same_teacher')}


def config(seed, arm, phase, pool, manifest_hash):
    if pool != matrix()[arm] or (phase != 'smoke' and arm not in ARMS):
        raise ValueError('Unplanned collaboration arm')
    result = B.config(seed, arm, phase, pool['teachers'], manifest_hash)
    result.pop('ensemble_rule')  # No teacher input-view ensemble in this experiment.
    result.update(pool=pool, pool_sha256=digest(pool), collaboration_rule=COLLABORATION_RULE)
    return result


def instrument(source, original_instrument):
    source = B.instrument(source, original_instrument)
    replacements = [
        ('queries_cat[:, queries.shape[1]:], args.offline_teacher_policy)',
         'queries_cat[:, queries.shape[1]:], args.offline_teacher_policy, teacher_models)'),
        ('            seq_loss = sequence_loss(\n                coord_predictions,',
         '            seq_loss = __coordinate_loss__(sequence_loss,\n                coord_predictions,'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Collaboration trainer anchor changed: ' + before)
        source = source.replace(before, after)
    return source


@contextmanager
def isolated_rng():
    """Extra teacher inference must not change student or next-step randomness."""
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def local_seed(queries, slot):
    from cotracker.utils.offline_teacher_policy import tensor_sha
    return int(hashlib.sha256((tensor_sha(queries) + ':' + str(slot)).encode()).hexdigest()[:15], 16)


def support_for(slot, video, queries):
    from cotracker.models.core.model_utils import get_points_on_a_grid
    if slot < 2:
        xy = get_points_on_a_grid(5, video.shape[-2:], device=video.device)
        return torch.cat((torch.zeros_like(xy[..., :1]), xy), -1)
    # A local generator, not the global augmentation/teacher/student RNG.
    generator = torch.Generator(device='cpu').manual_seed(local_seed(queries, slot))
    draw = torch.rand(1, 100, 3, generator=generator, dtype=torch.float32)
    draw[..., 0] = (draw[..., 0] * video.shape[1]).floor()
    draw[..., 1] *= video.shape[-1] - 1
    draw[..., 2] *= video.shape[-2] - 1
    return draw.to(video.device)


def supported(output, queries, height, width):
    p, v, c = output
    after = torch.arange(p.shape[1], device=p.device)[None, :, None] > queries[..., 0][:, None, :]
    # V is used as the existing validity filter, not as a calibrated probability.
    return (after & torch.isfinite(p).all(-1) & torch.isfinite(v) & (v > .9) &
            (p[..., 0] >= 0) & (p[..., 0] <= width - 1) &
            (p[..., 1] >= 0) & (p[..., 1] <= height - 1))


class Coordinator:
    def __init__(self, arm, pool):
        if pool != matrix()[arm]:
            raise ValueError('Wrong teacher pool')
        self.arm, self.pool, self.context = arm, pool, None

    def dispatch(self, slot, teacher_type, model, video, queries, draw, arm, teachers):
        from cotracker.utils.offline_teacher_policy import tensor_sha
        if arm != self.arm or len(teachers) != 3 or teachers[slot][1] is not model:
            raise ValueError('Teacher identity changed')
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        start = time.perf_counter()
        p, v, c, audit = B.teacher_forward(slot, teacher_type, model, video, queries, draw,
                                          'control', self.pool['teachers'])
        if not all(bool(torch.isfinite(x).all()) for x in (p, v, c)):
            raise ValueError('Nonfinite primary teacher label')
        outputs, evidence = [], []
        if arm == 'same_teacher':
            outputs = [(p, v, c), (p, v, c)]
        elif arm != 'control':
            with isolated_rng():
                for other in range(3):
                    if other == slot:
                        continue
                    typ, teacher = teachers[other]
                    r = B.teacher_forward(other, typ, teacher, video, queries, support_for(other, video, queries),
                                          'control', self.pool['teachers'])
                    outputs.append(r[:3])
                    evidence.append({'slot': other, **{k: r[3][k] for k in
                        (*LABEL_KEYS, 'support_draw_sha256', 'teacher_queries_sha256', 'teacher_seconds')}})
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        audit.update(policy=arm, pool_sha256=digest(self.pool), teacher_calls=1 + len(evidence),
                     teacher_seconds=time.perf_counter() - start,
                     collaboration={'arm': arm, 'reviewers': evidence, 'student_forward_passes': 1,
                                    'rule_sha256': digest(COLLABORATION_RULE)})
        self.context = {'primary': (p, v, c), 'reviewers': outputs, 'queries': queries,
                        'height': video.shape[-2], 'width': video.shape[-1], 'slot': slot,
                        'audit': audit['collaboration']}
        return p, v, c, audit

    def loss(self, original, predictions, targets, valids, vis=None, gamma=.8,
             add_huber_loss=False, loss_only_for_visible=False):
        """One student forward; unchanged primary V/C and other loss components."""
        from cotracker.models.core.cotracker.losses import huber_loss
        from cotracker.models.core.model_utils import reduce_masked_mean
        from cotracker.utils.offline_teacher_policy import tensor_sha
        ctx = self.context
        if ctx is None or len(targets) != 1 or not add_huber_loss or not loss_only_for_visible:
            raise ValueError('Only the fixed offline visible-coordinate loss may be adapted')
        p, v, c = ctx['primary']
        if not torch.equal(targets[0], p) or not torch.equal(vis[0], (v > .9).float()):
            raise ValueError('Primary label mapping changed')
        mask = valids[0].clone() * vis[0]
        if p.shape[0] != 1 or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Coverage-matched review requires batch one and binary validity')
        active = mask > 0
        valid = supported((p, v, c), ctx['queries'], ctx['height'], ctx['width']) & active
        weight, fused = torch.ones_like(mask), p.clone()
        stats = ctx['audit']
        stats.update(point_frames=p.shape[1] * p.shape[2], active=int(active.sum()),
                     eligible=0, conflict=0, downweighted=0, fused=0, missing_review=0,
                     disagreement_sum=0., disagreement_count=0, max_correction=0.,
                     downweight_source='none', base_weight_mass=float(mask.sum()))
        reviewers, rvalid = ctx['reviewers'], []
        conflict = torch.zeros_like(active)
        if reviewers:
            rvalid = [supported(r, ctx['queries'], ctx['height'], ctx['width']) & valid for r in reviewers]
            joint = rvalid[0] & rvalid[1]
            a, b = reviewers[0][0], reviewers[1][0]
            # Invalid coordinates cannot leak NaNs through an unselected branch.
            a = torch.where(rvalid[0][..., None], a, p)
            b = torch.where(rvalid[1][..., None], b, p)
            da, db, dab = (a - p).norm(dim=-1), (b - p).norm(dim=-1), (a - b).norm(dim=-1)
            threshold = 4. if self.arm == 'gate4' else 2.
            conflict = joint & (dab <= threshold) & (da > threshold) & (db > threshold)
            agreement = joint & (dab <= threshold) & (da <= threshold) & (db <= threshold)
            stats.update(eligible=int(joint.sum()), conflict=int(conflict.sum()),
                         missing_review=int((valid & ~joint).sum()),
                         disagreement_sum=float((da[joint] + db[joint]).sum()),
                         disagreement_count=2 * int(joint.sum()))
            if self.arm in ('gate2', 'gate4', 'fuse2', 'shuffle2'):
                down = conflict.clone()
                if self.arm == 'shuffle2':
                    down.zero_()
                    generator = torch.Generator(device='cpu').manual_seed(local_seed(ctx['queries'], ctx['slot']) + 1)
                    for t in range(p.shape[1]):
                        # Match downweight count per frame within the same review-eligible population.
                        ids = torch.where(joint[0, t])[0]
                        n = int(conflict[0, t].sum())
                        perm = torch.randperm(len(ids), generator=generator).to(ids.device)
                        down[0, t, ids[perm[:n]]] = True
                    if not torch.equal(down.sum(-1), conflict.sum(-1)):
                        raise ValueError('Shuffled coverage mismatch')
                weight[down] = .25
                stats.update(downweighted=int(down.sum()),
                             downweight_source='shuffled' if self.arm == 'shuffle2' else 'reviewer_agreement')
                if self.arm == 'fuse2':
                    correction = .25 * ((a + b) * .5 - p)
                    norm = correction.norm(dim=-1, keepdim=True)
                    correction *= torch.clamp(.5 / norm.clamp_min(1e-12), max=1.)
                    fused = torch.where(agreement[..., None], p + correction, p)
                    change = (fused - p).norm(dim=-1)
                    stats.update(fused=int((change > 0).sum()), max_correction=float(change.max()))
        stats.update(weight_mass=float((mask * weight).sum()),
                     supervision_sha256=digest({'target': tensor_sha(fused), 'weights': tensor_sha(weight),
                                               'reviewers': [tensor_sha(r[0]) for r in reviewers] if self.arm == 'tutor010' else []}),
                     weights_sha256=tensor_sha(weight), target_sha256=tensor_sha(fused),
                     conflict_sha256=tensor_sha(conflict))
        if self.arm in NOOPS:
            # Exact no-op path, including arithmetic order and original denominator.
            return original(predictions, targets, valids, vis=vis, gamma=gamma,
                            add_huber_loss=add_huber_loss, loss_only_for_visible=loss_only_for_visible)
        losses = []
        for i, pred in enumerate(predictions[0]):
            values = huber_loss(pred, fused, delta=6.).mean(-1)
            if self.arm == 'tutor010':
                base = values
                for r, ok in zip(reviewers, rvalid):
                    safe = torch.where(ok[..., None], r[0], p)
                    other = huber_loss(pred, safe, delta=6.).mean(-1)
                    values = values + .05 * torch.where(ok, other - base, torch.zeros_like(base))
            losses.append(gamma ** (len(predictions[0]) - i - 1) * reduce_masked_mean(values, mask * weight))
        return sum(losses) / len(losses)


def records(path, cfg):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        i, a = r['step'], r['label_audit']
        slot, arm = r['primary_index'], cfg['policy']
        if type(i) is not int or not 0 <= i < cfg['steps'] or slot not in range(3):
            raise ValueError('Invalid step')
        if (r['architecture'] != 'base' or r['auxiliary_index'] is not None or r['teacher_policy'] != arm or
            a['policy'] != arm or a['pool_sha256'] != cfg['pool_sha256'] or
            a['teacher_type'] != TEACHER_TYPES[slot] or a['teacher_setting'] != B.baseline()[slot]):
            raise ValueError('Teacher configuration mismatch')
        q = a['collaboration']
        if q['rule_sha256'] != digest(COLLABORATION_RULE) or q['arm'] != arm or q['student_forward_passes'] != 1:
            raise ValueError('Collaboration rule changed')
        expected = [] if arm in ('control', 'same_teacher') else [s for s in range(3) if s != slot]
        if [x['slot'] for x in q['reviewers']] != expected or a['teacher_calls'] != 1 + len(expected):
            raise ValueError('Review coverage changed')
        if (q['point_frames'] != 7680 or not 0 <= q['conflict'] <= q['eligible'] <= q['active'] <= 7680 or
            not 0 <= q['fused'] <= q['eligible'] or not 0 <= q['max_correction'] <= .5001 or
            not 0 <= q['weight_mass'] <= q['base_weight_mass'] or q['weight_mass'] < .25 * q['base_weight_mass'] - .001):
            raise ValueError('Invalid coordinate coverage/correction')
        if arm in ('gate2', 'gate4', 'fuse2', 'shuffle2'):
            if q['downweighted'] != q['conflict']:
                raise ValueError('Review weight count changed')
        elif q['downweighted'] or q['weight_mass'] != q['base_weight_mass']:
            raise ValueError('Unexpected reweighting')
        if arm != 'fuse2' and (q['fused'] or q['target_sha256'] != a['tracks_sha256']):
            raise ValueError('Unexpected coordinate correction')
        for k in LABEL_KEYS:
            if a[k] != a['base_' + k]:
                raise ValueError('Primary P/V/C changed')
        for value in [*(a[k] for k in (*LABEL_KEYS, 'support_draw_sha256')),
                      *(q[k] for k in ('supervision_sha256', 'weights_sha256', 'target_sha256', 'conflict_sha256')),
                      *(x[k] for x in q['reviewers'] for k in (*LABEL_KEYS, 'support_draw_sha256', 'teacher_queries_sha256'))]:
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid audit hash')
        if len(a['query_time_counts']) != 60 or sum(a['query_time_counts']) != 128:
            raise ValueError('Query coverage changed')
        if any(not math.isfinite(r[k]) for k in ('loss', 'step_seconds')) or r['step_seconds'] <= 0:
            raise ValueError('Invalid loss/timing')
        if any(not math.isfinite(x['teacher_seconds']) or x['teacher_seconds'] < 0 for x in q['reviewers']):
            raise ValueError('Invalid reviewer timing')
        h = r['health']
        if any(not math.isfinite(x) or x < 0 for x in h.values()) or max(h['peak_allocated_mib'], h['peak_reserved_mib']) > 20480 or h['driver_free_mib'] < 2048:
            raise ValueError('GPU headroom exceeded')
        signature = lambda z: [z[k] for k in INPUT_KEYS] + [z['label_audit'][k] for k in (*LABEL_KEYS, 'support_draw_sha256')] + [
            z['label_audit']['collaboration'][k] for k in ('supervision_sha256', 'weights_sha256', 'target_sha256', 'conflict_sha256')] + [
            [[x[k] for k in ('slot', *LABEL_KEYS, 'support_draw_sha256', 'teacher_queries_sha256')]
             for x in z['label_audit']['collaboration']['reviewers']]]
        if i in rows and signature(rows[i]) != signature(r):
            raise ValueError('Resumed teacher supervision changed')
        rows[i] = r
    if set(rows) != set(range(cfg['steps'])) or statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2) > 8:
        raise ValueError('Incomplete or slow training')
    return rows


def paired(reference, candidate, reference_pool, candidate_pool):
    if set(reference) != set(candidate):
        raise ValueError('Pair coverage differs')
    changed = [0, 0, 0]
    for i, row in candidate.items():
        ref = reference[i]
        if any(ref[k] != row[k] for k in INPUT_KEYS) or any(ref['label_audit'][k] != row['label_audit'][k]
            for k in (*LABEL_KEYS, 'support_draw_sha256', 'query_time_counts')):
            raise ValueError('Primary input/label/RNG differs')
        changed[row['primary_index']] += int(ref['label_audit']['collaboration']['supervision_sha256'] !=
                                              row['label_audit']['collaboration']['supervision_sha256'])
    return changed
