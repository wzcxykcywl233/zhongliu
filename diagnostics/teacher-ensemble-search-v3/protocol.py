"""Finite, validation-only search over teacher slots; student recipe stays fixed."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

BASE_SEED = 20261008
SEEDS = (0, 1, 2)
STEPS = {'smoke': 10, 'pilot': 250, 'full': 1000}
PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
RUNTIME = {'backend': 'compact', 'point_chunk_size': 0, 'gain': 1.,
           'max_correction_pixels': 0., 'checkpoint_blocks': False}
FAMILIES = ('v2', 'on3', 'offold', 'offscaled')
FAMILY_SLOT = {'v2': 0, 'on3': 1, 'offold': 2, 'offscaled': 2}
FIELDS = ('DSC', 'HD95', 'MASD', 'CD', 'D98')
RULE = {'minimum_dsc_gain': .00001, 'maximum_d98_drop': .002,
        'maximum_hd95_increase': .02, 'minimum_improved_seeds_full': 2,
        'maximum_single_seed_dsc_drop': .002}
INPUT_KEYS = ('case_names', 'queries_sha256', 'video_sha256', 'primary_index', 'auxiliary_index')
LABEL_KEYS = ('tracks_sha256', 'visibility_sha256', 'confidence_sha256')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


TRANSFORMS = ('xmean', 'xguard', 'xyguard')
SMOKE_ARMS = ('control', 'v2_xmean', 'on3_xguard', 'offold_xyguard', 'offscaled_xguard')
ENSEMBLE_RULE = {'maximum_disagreement_pixels': 2., 'minimum_visibility': .9,
                 'minimum_confidence': .5, 'preserve_original_vc': True,
                 'preserve_query_frames': True, 'online_only_after_query': True}


def view_modes(mode):
    if mode == 'none':
        return ('none',)
    if mode in ('xmean', 'xguard'):
        return ('none', 'flip_x')
    if mode == 'xyguard':
        return ('none', 'flip_x', 'flip_y')
    raise ValueError('Unknown ensemble mode')


def baseline():
    # v1 validation-selected on3_s0, not the historical online-25 control.
    return [{'family': 'v2', 'checkpoint': 'cotracker2v1.pth', 'support': 5, 'iterations': 4, 'transform': 'none'},
            {'family': 'on3', 'checkpoint': 'baseline_online.pth', 'support': 0, 'iterations': 4, 'transform': 'none'},
            {'family': 'offscaled', 'checkpoint': 'scaled_offline.pth', 'support': 100, 'iterations': 4, 'transform': 'none'}]


def matrix():
    pools = {'control': baseline()}
    pools['oldpool'] = baseline()
    pools['oldpool'][2].update(family='offold', checkpoint='baseline_offline.pth')
    groups = {}
    for family in FAMILIES:
        parent = 'oldpool' if family == 'offold' else 'control'
        groups[family] = [parent]
        for transform in TRANSFORMS:
            name = family + '_' + transform
            pool = copy.deepcopy(pools[parent])
            pool[FAMILY_SLOT[family]]['transform'] = transform
            pools[name] = pool
            groups[family].append(name)
    return pools, groups


def config(seed, arm, phase, pool, manifest_hash):
    return {'kind': 'mamba', 'policy': arm, 'phase': phase, 'pool': pool, 'pool_sha256': digest(pool),
            'offline_checkpoint': pool[2]['checkpoint'], 'seed': seed, 'training_seed': BASE_SEED + seed,
            'steps': STEPS[phase], 'scheduler_num_steps': 1000, 'sequence_len': 60, 'traj_per_sample': 128,
            'train_iters': 4, 'lr': .00005, 'confidence_target_mode': 'hard', 'auxiliary_teacher_weight': 0,
            'mask_loss_weight': 0, 'sidecar_runtime': RUNTIME, 'ensemble_rule': ENSEMBLE_RULE,
            'manifest_sha256': manifest_hash}


def means(samples):
    return {k: statistics.mean(float(r[k]) for r in samples) for k in FIELDS}


def admissible(samples, reference):
    return selection_evidence(samples, reference)['Eligible']


def selection_evidence(samples, reference):
    if len(samples) != len(reference) or len(samples) not in (1, 3):
        raise ValueError('Unpaired selection samples')
    if any(not math.isfinite(float(r[k])) for r in samples + reference for k in FIELDS):
        raise ValueError('Nonfinite selection metric')
    a, b = means(samples), means(reference)
    changes = [s['DSC'] - r['DSC'] for s, r in zip(samples, reference)]
    checks = {'DSC_gain': a['DSC'] > b['DSC'] + RULE['minimum_dsc_gain'],
              'D98_drop': a['D98'] >= b['D98'] - RULE['maximum_d98_drop'],
              'HD95_increase': a['HD95'] <= b['HD95'] + RULE['maximum_hd95_increase'],
              'improved_seeds': len(samples) == 1 or sum(x > 0 for x in changes) >= RULE['minimum_improved_seeds_full'],
              'worst_seed': len(samples) == 1 or min(changes) >= -RULE['maximum_single_seed_dsc_drop']}
    return {'N': len(samples), **{'Delta_' + k: a[k] - b[k] for k in FIELDS},
            'ImprovedSeeds': sum(x > 0 for x in changes), 'WorstSeedDSC': min(changes),
            'Eligible': all(checks.values()), 'FailedRules': ';'.join(k for k, v in checks.items() if not v)}


def best(scores, names, reference='control'):
    eligible = [p for p in names if p != reference and admissible(scores[p], scores[reference])]
    if not eligible:
        return reference
    return min(eligible, key=lambda p: (-means(scores[p])['DSC'], means(scores[p])['HD95'],
                                      means(scores[p])['CD'], -means(scores[p])['D98'], p))


def combinations(pools, winners):
    """Include every subset of accepted slot changes; reuse identical full arms."""
    found = {'control': pools['control']}
    choices = [(slot, name) for slot, name in sorted(winners.items()) if name != 'control']
    identities = {digest(pool): name for name, pool in pools.items()}
    for size in range(1, len(choices) + 1):
        for subset in itertools.combinations(choices, size):
            pool = copy.deepcopy(pools['control'])
            for slot, name in subset:
                pool[slot] = copy.deepcopy(pools[name][slot])
            name = identities.get(digest(pool), 'mix' + ''.join(str(s) for s, _ in subset))
            found[name] = pool
    return found


def instrument(source, original_instrument):
    source = original_instrument(source)
    begin = "    if teacher_study:\n        torch.cuda.synchronize(video.device)\n        teacher_started = time.perf_counter()"
    end = "    vis_g = vis_g > 0.9"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('Teacher dispatch anchors changed')
    a, b = source.index(begin), source.index(end)
    source = source[:a] + (
        "    if not teacher_study:\n        raise ValueError('Teacher pool search requires paired study')\n"
        "    trajs_g, vis_g, confidence, batch.teacher_label_audit = __dispatch__(\n"
        "        teacher_model_ind, teacher_model_type, teacher_model, video, queries,\n"
        "        queries_cat[:, queries.shape[1]:], args.offline_teacher_policy)\n\n") + source[b:]
    for before, after in (
        ("choices=OFFLINE_POLICIES, default='old'", "choices=tuple(__settings__), default='control'"),
        ("expected = OFFLINE_SETTINGS[args.offline_teacher_policy][0]", "expected = __settings__[args.offline_teacher_policy]"),
        ("final['teacher_student_policy'] = args.offline_teacher_policy", "final['teacher_student_policy'] = args.offline_teacher_policy\n                    final['teacher_pool_sha256'] = __pool_sha__")):
        if source.count(before) != 1:
            raise ValueError('Trainer adaptation anchor changed: ' + before)
        source = source.replace(before, after)
    return source


def transform_inputs(video, queries, support, mode):
    """No interpolation, RNG or in-place edits. One view of one teacher."""
    import torch
    if mode not in ('none', 'flip_x', 'flip_y'):
        raise ValueError('Unknown teacher transform')
    if video.dtype != torch.float32 or video.ndim != 5 or video.shape[0] != 1:
        raise ValueError('Teacher video must be float32 B/T/C/H/W with B=1')
    if not bool(torch.isfinite(video).all()) or video.min() < 0 or video.max() > 255:
        raise ValueError('Teacher video must be finite in [0,255]')
    meta = {'mode': mode, 'fallback': False}
    if mode == 'none':
        return video, queries, support, meta
    if mode in ('flip_x', 'flip_y'):
        axis, coordinate = (-1, 1) if mode == 'flip_x' else (-2, 2)
        q, s = queries.clone(), support.clone()
        q[..., coordinate] = video.shape[axis] - 1 - q[..., coordinate]
        s[..., coordinate] = video.shape[axis] - 1 - s[..., coordinate]
        return video.flip(axis), q, s, meta


def inverse_tracks(tracks, mode, height, width):
    if mode not in ('flip_x', 'flip_y'):
        return tracks
    result = tracks.clone()
    axis, extent = (0, width) if mode == 'flip_x' else (1, height)
    result[..., axis] = extent - 1 - result[..., axis]
    return result


def fuse_views(outputs, queries, mode, height, width, offline):
    """Fuse positions only; disagreement is evidence, not correspondence truth."""
    import torch
    if len(outputs) != len(view_modes(mode)):
        raise ValueError('Wrong ensemble view count')
    p0, v0, c0 = outputs[0]
    if (p0.ndim != 4 or p0.shape[0] != 1 or p0.shape[-1] != 2 or
        queries.shape != (1, p0.shape[2], 3) or any(x.dtype != torch.float32 for x in (p0, v0, c0)) or
        v0.shape != p0.shape[:-1] or c0.shape != v0.shape or
        any(not bool(torch.isfinite(x).all()) for x in (p0, v0, c0))):
        raise ValueError('Invalid original teacher outputs')
    frames = torch.arange(p0.shape[1], device=p0.device)[None, :, None]
    eligible = (frames != queries[:, None, :, 0]) if offline else (frames > queries[:, None, :, 0])
    total = p0.shape[1] * p0.shape[2]
    stats = {'point_frames': total, 'eligible_point_frames': int(eligible.sum()),
             'accepted_point_frames': 0, 'changed_point_frames': 0, 'fallback_point_frames': 0,
             'mean_disagreement_pixels': 0., 'mean_correction_pixels': 0., 'max_correction_pixels': 0.}
    if mode == 'none':
        return p0, v0, c0, stats
    valid = eligible.clone()
    for p, v, c in outputs:
        if (p.shape != p0.shape or v.shape != v0.shape or c.shape != c0.shape or
            any(x.dtype != torch.float32 or x.device != p0.device for x in (p, v, c))):
            raise ValueError('Ensemble shape/device/dtype changed')
        valid &= (torch.isfinite(p).all(-1) & torch.isfinite(v) & torch.isfinite(c) &
                  (p[..., 0] >= 0) & (p[..., 0] <= width - 1) & (p[..., 1] >= 0) & (p[..., 1] <= height - 1))
        if mode != 'xmean':
            valid &= (v > ENSEMBLE_RULE['minimum_visibility']) & (c >= ENSEMBLE_RULE['minimum_confidence'])
    disagreement = torch.zeros_like(v0)
    for a, b in itertools.combinations(outputs, 2):
        disagreement = torch.maximum(disagreement, torch.linalg.vector_norm(a[0] - b[0], dim=-1))
    if mode != 'xmean':
        valid &= disagreement <= ENSEMBLE_RULE['maximum_disagreement_pixels']
    # Invalid alternate values must not contaminate fallback positions.
    average = p0 + torch.stack([p - p0 for p, _, _ in outputs]).mean(0)
    p = torch.where(valid[..., None], average, p0)
    correction = torch.linalg.vector_norm(p - p0, dim=-1)
    defined = eligible & torch.isfinite(disagreement)
    stats.update(accepted_point_frames=int(valid.sum()), changed_point_frames=int((correction > 0).sum()),
                 fallback_point_frames=int((eligible & ~valid).sum()),
                 mean_disagreement_pixels=float(disagreement[defined].mean()) if bool(defined.any()) else 0.,
                 mean_correction_pixels=float(correction[eligible].mean()) if bool(eligible.any()) else 0.,
                 max_correction_pixels=float(correction.max()))
    return p, v0, c0, stats


def teacher_forward(slot, teacher_type, model, video, queries, draw, arm, pool):
    import time
    import torch
    from cotracker.utils.offline_teacher_policy import label_audit, tensor_sha
    setting = pool[slot]
    if (teacher_type != ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot] or
        setting['family'] not in (('v2',), ('on3',), ('offold', 'offscaled'))[slot] or
        model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Wrong teacher slot or unfrozen teacher')
    if queries.ndim != 3 or queries.shape[0] != 1 or queries.shape[-1] != 3 or draw.shape != (1, 25 if slot < 2 else 100, 3):
        raise ValueError('Original support draw/query shape changed')
    if not all(bool(torch.isfinite(x).all()) for x in (queries, draw)):
        raise ValueError('Nonfinite teacher input')
    t = queries[..., 0]
    if bool(((t < 0) | (t >= video.shape[1]) | (t != t.round())).any()):
        raise ValueError('Invalid arbitrary query time')
    if setting['iterations'] != 4 or setting['support'] != (5, 0, 100)[slot]:
        raise ValueError('Input experiment must preserve on3_s0 support/iterations')
    support = draw if slot != 1 else draw[:, :0]
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    start = time.perf_counter()
    n = queries.shape[1]
    outputs, evidence = [], []
    modes = view_modes(setting['transform'])
    for mode in modes:
        view, q, s, _ = transform_inputs(video, queries, support, mode)
        combined = torch.cat((q, s), 1) if s.shape[1] else q
        with torch.no_grad():
            result = model(view, combined, iters=4)
        p = inverse_tracks(result[0][:, :, :n], mode, video.shape[-2], video.shape[-1])
        v = result[1][:, :, :n]
        c = result[2][:, :, :n] if slot else torch.ones_like(v)
        outputs.append((p, v, c))
        stride = max(1, math.ceil(view.numel() / 65536))
        evidence.append({'mode': mode, 'queries_sha256': tensor_sha(combined),
                         'video_sample_sha256': tensor_sha(view.reshape(-1)[::stride]),
                         'tracks_sha256': tensor_sha(p), 'visibility_sha256': tensor_sha(v),
                         'confidence_sha256': tensor_sha(c), 'video_sample_stride': stride})
    p, v, c, stats = fuse_views(outputs, queries, setting['transform'], video.shape[-2], video.shape[-1], slot == 2)
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    seconds = time.perf_counter() - start
    audit = label_audit(video, queries, p, v, c, policy='scaled', teacher_type=teacher_type,
                        support_draw_sha256=tensor_sha(draw), seconds=seconds)
    audit.update(policy=arm, teacher_setting=setting, pool_sha256=digest(pool),
                 support_used=support.shape[1], iterations=4,
                 transform={'mode': setting['transform'], 'fallback': bool(stats['fallback_point_frames'])},
                 teacher_calls=len(modes), teacher_queries_sha256=evidence[0]['queries_sha256'],
                 teacher_video_sample_sha256=evidence[0]['video_sample_sha256'],
                 output_coordinates='original', views=evidence, ensemble=stats,
                 ensemble_rule=ENSEMBLE_RULE,
                 base_tracks_sha256=evidence[0]['tracks_sha256'],
                 base_visibility_sha256=evidence[0]['visibility_sha256'],
                 base_confidence_sha256=evidence[0]['confidence_sha256'])
    return p, v, c, audit


def records(path, cfg):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        i, a, slot = r['step'], r['label_audit'], r['primary_index']
        if type(i) is not int or not 0 <= i < cfg['steps'] or slot not in (0, 1, 2):
            raise ValueError('Invalid step or teacher slot')
        setting = cfg['pool'][slot]
        if (r['teacher_policy'] != cfg['policy'] or r['architecture'] != 'base' or r['auxiliary_index'] is not None or
            a['policy'] != cfg['policy'] or a['pool_sha256'] != cfg['pool_sha256'] or a['teacher_setting'] != setting or
            a['teacher_type'] != ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot] or
            a['support_used'] != (setting['support'] ** 2 if slot < 2 else setting['support']) or
            a['iterations'] != setting['iterations']):
            raise ValueError('Teacher settings not executed')
        modes = view_modes(setting['transform'])
        if (a['transform']['mode'] != setting['transform'] or a['teacher_calls'] != len(modes) or
            a['output_coordinates'] != 'original' or type(a['transform']['fallback']) is not bool):
            raise ValueError('Teacher transform evidence changed')
        if (a['ensemble_rule'] != ENSEMBLE_RULE or tuple(x['mode'] for x in a['views']) != modes or
            any(a['base_' + k] != a['views'][0][k] for k in LABEL_KEYS) or
            a['visibility_sha256'] != a['base_visibility_sha256'] or
            a['confidence_sha256'] != a['base_confidence_sha256']):
            raise ValueError('Original teacher V/C or ensemble evidence changed')
        stats = a['ensemble']
        counts = [stats[k] for k in ('point_frames', 'eligible_point_frames', 'accepted_point_frames', 'changed_point_frames', 'fallback_point_frames')]
        if (any(type(x) is not int or x < 0 for x in counts) or counts[0] != 60 * 128 or
            not counts[3] <= counts[2] <= counts[1] <= counts[0] or
            (setting['transform'] != 'none' and counts[2] + counts[4] != counts[1]) or
            any(not math.isfinite(stats[k]) or stats[k] < 0 for k in
                ('mean_disagreement_pixels', 'mean_correction_pixels', 'max_correction_pixels'))):
            raise ValueError('Invalid ensemble counts/corrections')
        if a['transform']['fallback'] != bool(counts[4]):
            raise ValueError('Fallback evidence disagrees with counts')
        if not counts[3] and a['tracks_sha256'] != a['base_tracks_sha256']:
            raise ValueError('Zero corrections changed label hash')
        if setting['transform'] in ('xguard', 'xyguard') and stats['max_correction_pixels'] > 2 * (len(modes) - 1) / len(modes) + .001:
            raise ValueError('Guarded correction exceeds pairwise bound')
        if setting['transform'] == 'none' and (counts[2] or counts[3] or counts[4] or a['tracks_sha256'] != a['base_tracks_sha256']):
            raise ValueError('No-op ensemble changed labels')
        for view in a['views']:
            for key in ('queries_sha256', 'video_sample_sha256', *LABEL_KEYS):
                value = view[key]
                if len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
                    raise ValueError('Invalid view hash')
        for key in ('teacher_queries_sha256', 'teacher_video_sample_sha256'):
            value = a[key]
            if len(value) != 64 or any(x not in '0123456789abcdef' for x in value):
                raise ValueError('Invalid transform evidence hash')
        if (a['teacher_queries_sha256'] != a['views'][0]['queries_sha256'] or
            a['teacher_video_sample_sha256'] != a['views'][0]['video_sample_sha256']):
            raise ValueError('Original view input hashes disagree')
        counts = a['query_time_counts']
        if len(counts) != 60 or sum(counts) != 128 or any(type(v) is not int or v < 0 for v in counts):
            raise ValueError('Training queries changed')
        for k in (*LABEL_KEYS, 'support_draw_sha256'):
            if len(a[k]) != 64 or any(x not in '0123456789abcdef' for x in a[k]):
                raise ValueError('Invalid label hash')
        if any(not math.isfinite(r[k]) for k in ('loss', 'step_seconds')) or r['step_seconds'] <= 0:
            raise ValueError('Invalid training loss/timing')
        if not math.isfinite(a['teacher_seconds']) or a['teacher_seconds'] < 0:
            raise ValueError('Invalid teacher timing')
        if any(not 0 <= a[k] <= 1 for k in ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction')):
            raise ValueError('Invalid label diagnostic')
        h = r['health']
        if any(not math.isfinite(h[k]) for k in ('peak_allocated_mib', 'peak_reserved_mib', 'driver_free_mib')) or min(h.values()) < 0:
            raise ValueError('Invalid memory evidence')
        if max(h['peak_allocated_mib'], h['peak_reserved_mib']) > 20480 or h['driver_free_mib'] < 2048:
            raise ValueError('GPU memory limit exceeded')
        sig = lambda z: [z[k] for k in INPUT_KEYS] + [z['label_audit'][k] for k in (*LABEL_KEYS, 'support_draw_sha256', 'query_time_counts', 'teacher_queries_sha256', 'teacher_video_sample_sha256', 'views', 'ensemble')]
        if i in rows and sig(rows[i]) != sig(r):
            raise ValueError('Resumed inputs or labels changed')
        rows[i] = r
    if set(rows) != set(range(cfg['steps'])) or statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2) > 8:
        raise ValueError('Missing steps or training too slow')
    return rows


def paired(reference, candidate, reference_pool, candidate_pool):
    if set(reference) != set(candidate):
        raise ValueError('Pair coverage changed')
    changed = [0, 0, 0]
    for i, row in candidate.items():
        ref = reference[i]
        if any(ref[k] != row[k] for k in INPUT_KEYS) or any(ref['label_audit'][k] != row['label_audit'][k] for k in ('support_draw_sha256', 'query_time_counts')):
            raise ValueError('Paired input/RNG changed')
        slot = row['primary_index']
        different = any(ref['label_audit'][k] != row['label_audit'][k] for k in LABEL_KEYS)
        if reference_pool[slot] == candidate_pool[slot] and different:
            raise ValueError('Unmodified teacher labels changed')
        base = lambda s: {k: v for k, v in s.items() if k != 'transform'}
        if base(reference_pool[slot]) == base(candidate_pool[slot]):
            if any(ref['label_audit']['base_' + k] != row['label_audit']['base_' + k] for k in LABEL_KEYS):
                raise ValueError('Original teacher labels changed during ensemble')
        changed[slot] += int(different)
    for slot in range(3):
        if reference_pool[slot]['checkpoint'] != candidate_pool[slot]['checkpoint'] and not changed[slot]:
            raise ValueError('Changed teacher produced no changed labels')
    return changed
