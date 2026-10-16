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


def baseline():
    return [{'family': 'v2', 'checkpoint': 'cotracker2v1.pth', 'support': 5, 'iterations': 4},
            {'family': 'on3', 'checkpoint': 'baseline_online.pth', 'support': 5, 'iterations': 4},
            {'family': 'offscaled', 'checkpoint': 'scaled_offline.pth', 'support': 100, 'iterations': 4}]


def matrix():
    pools = {'control': baseline()}
    pools['oldpool'] = baseline()
    pools['oldpool'][2].update(family='offold', checkpoint='baseline_offline.pth')
    groups = {}
    for family in FAMILIES:
        parent = 'oldpool' if family == 'offold' else 'control'
        groups[family] = [parent]
        for label, field, value in (('s0', 'support', 0), ('slo', 'support', 3 if FAMILY_SLOT[family] < 2 else 50),
                                    ('i2', 'iterations', 2), ('i6', 'iterations', 6), ('i8', 'iterations', 8)):
            name = family + '_' + label
            pool = copy.deepcopy(pools[parent])
            pool[FAMILY_SLOT[family]][field] = value
            pools[name] = pool
            groups[family].append(name)
    return pools, groups


def config(seed, arm, phase, pool, manifest_hash):
    return {'kind': 'mamba', 'policy': arm, 'phase': phase, 'pool': pool, 'pool_sha256': digest(pool),
            'offline_checkpoint': pool[2]['checkpoint'], 'seed': seed, 'training_seed': BASE_SEED + seed,
            'steps': STEPS[phase], 'scheduler_num_steps': 1000, 'sequence_len': 60, 'traj_per_sample': 128,
            'train_iters': 4, 'lr': .00005, 'confidence_target_mode': 'hard', 'auxiliary_teacher_weight': 0,
            'mask_loss_weight': 0, 'sidecar_runtime': RUNTIME, 'manifest_sha256': manifest_hash}


def means(samples):
    return {k: statistics.mean(float(r[k]) for r in samples) for k in FIELDS}


def admissible(samples, reference):
    if len(samples) != len(reference) or len(samples) not in (1, 3):
        raise ValueError('Unpaired selection samples')
    if any(not math.isfinite(float(r[k])) for r in samples + reference for k in FIELDS):
        raise ValueError('Nonfinite selection metric')
    a, b = means(samples), means(reference)
    changes = [s['DSC'] - r['DSC'] for s, r in zip(samples, reference)]
    return (a['DSC'] > b['DSC'] + RULE['minimum_dsc_gain'] and
            a['D98'] >= b['D98'] - RULE['maximum_d98_drop'] and
            a['HD95'] <= b['HD95'] + RULE['maximum_hd95_increase'] and
            (len(samples) == 1 or (sum(x > 0 for x in changes) >= RULE['minimum_improved_seeds_full'] and
                                  min(changes) >= -RULE['maximum_single_seed_dsc_drop'])))


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


def teacher_forward(slot, teacher_type, model, video, queries, draw, arm, pool):
    import time
    import torch
    from cotracker.models.core.model_utils import get_points_on_a_grid
    from cotracker.utils.offline_teacher_policy import label_audit, tensor_sha
    setting = pool[slot]
    if (teacher_type != ('online', 'online_cotracker_three', 'offline_cotracker_three')[slot] or
        setting['family'] not in (('v2',), ('on3',), ('offold', 'offscaled'))[slot] or
        model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Wrong teacher slot or unfrozen teacher')
    if queries.ndim != 3 or queries.shape[0] != 1 or queries.shape[-1] != 3 or draw.shape != (1, 25 if slot < 2 else 100, 3):
        raise ValueError('Original support draw/query shape changed')
    if not all(bool(torch.isfinite(x).all()) for x in (video, queries, draw)):
        raise ValueError('Nonfinite teacher input')
    t = queries[..., 0]
    if bool(((t < 0) | (t >= video.shape[1]) | (t != t.round())).any()):
        raise ValueError('Invalid arbitrary query time')
    amount = setting['support']
    if slot < 2:
        if amount == 5:
            support = draw
        elif amount == 0:
            support = draw[:, :0]
        elif amount == 3:
            xy = get_points_on_a_grid(amount, video.shape[-2:], device=video.device)
            support = torch.cat((torch.zeros_like(xy[..., :1]), xy), -1)
        else:
            raise ValueError('Unsupported online support setting')
    else:
        if amount not in (0, 50, 100):
            raise ValueError('Unsupported offline support setting')
        support = draw[:, :amount]
    combined = torch.cat((queries, support), 1) if support.shape[1] else queries
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    start = time.perf_counter()
    with torch.no_grad():
        result = model(video, combined, iters=setting['iterations'])
    n = queries.shape[1]
    p, v = result[0][:, :, :n], result[1][:, :, :n]
    c = result[2][:, :, :n] if slot else torch.ones_like(v)
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    audit = label_audit(video, queries, p, v, c, policy='scaled', teacher_type=teacher_type,
                        support_draw_sha256=tensor_sha(draw), seconds=time.perf_counter() - start)
    audit.update(policy=arm, teacher_setting=setting, pool_sha256=digest(pool),
                 support_used=support.shape[1], iterations=setting['iterations'])
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
        sig = lambda z: [z[k] for k in INPUT_KEYS] + [z['label_audit'][k] for k in (*LABEL_KEYS, 'support_draw_sha256', 'query_time_counts')]
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
        changed[slot] += int(different)
    for slot in range(3):
        if reference_pool[slot] != candidate_pool[slot] and not changed[slot]:
            raise ValueError('Changed teacher produced no changed labels')
    return changed
