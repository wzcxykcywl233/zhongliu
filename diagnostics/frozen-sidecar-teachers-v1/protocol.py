"""Fixed frozen-backbone Mamba student; only the offline teacher changes."""
import hashlib
import json
import math
from pathlib import Path
import statistics

POLICIES = ('old', 'scaled', 'grid0', 'iter6')
SETTINGS = {'old': ('baseline_offline.pth', 100, 4),
            'scaled': ('scaled_offline.pth', 100, 4),
            'grid0': ('scaled_offline.pth', 0, 4),
            'iter6': ('scaled_offline.pth', 0, 6)}
SEEDS = (0, 1, 2)
BASE_SEED = 20261008
STEPS = 1000
SMOKE_STEPS = 10  # Original seed first samples the offline teacher at step 7.
PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
KIND = 'mamba'
PARAMETERS = 28763025
TRAINABLE = 3377325
RUNTIME = {'backend': 'compact', 'point_chunk_size': 0, 'gain': 1.,
           'max_correction_pixels': 0., 'checkpoint_blocks': False}
INPUT_KEYS = ('case_names', 'queries_sha256', 'video_sha256', 'primary_index', 'auxiliary_index')
LABEL_KEYS = ('tracks_sha256', 'visibility_sha256', 'confidence_sha256')
PAIRS = (('scaled', 'old'), ('grid0', 'scaled'), ('iter6', 'grid0'),
         ('grid0', 'old'), ('iter6', 'old')) + tuple((p, 'pretrained') for p in POLICIES)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def train_config(seed, policy, smoke, manifest_hash):
    return {'kind': KIND, 'policy': policy, 'offline_checkpoint': SETTINGS[policy][0],
            'seed': seed, 'training_seed': BASE_SEED + seed, 'steps': SMOKE_STEPS if smoke else STEPS,
            'scheduler_num_steps': STEPS, 'sequence_len': 60, 'traj_per_sample': 128,
            'train_iters': 4, 'lr': .00005, 'confidence_target_mode': 'hard',
            'auxiliary_teacher_weight': 0, 'mask_loss_weight': 0,
            'sidecar_runtime': RUNTIME, 'manifest_sha256': manifest_hash}


def instrument(source):
    """Narrow checked adaptation of the frozen trainer; optimizer/loss unchanged."""
    old = "args.feature_sidecar != 'none' or not args.trackrad_data_dir or args.batch_size != 1 or"
    new = "args.feature_sidecar != 'mamba' or not args.trackrad_data_dir or args.batch_size != 1 or"
    anchors = ((old, new, 1),
               ('teacher study requires paired base, no sidecar, TrackRAD, batch=1',
                'teacher study requires frozen base, Mamba sidecar, TrackRAD, batch=1', 1),
               ('if total_steps >= args.num_steps:', 'if total_steps >= __stop_steps__:', 2),
               ('                    training_step_started = time.perf_counter()',
                '                    training_step_started = time.perf_counter()\n'
                '                    torch.cuda.reset_peak_memory_stats()', 1),
               ("                        with open(Path(args.ckpt_path) / 'paired-steps.jsonl', 'a', encoding='utf-8') as stream:",
                "                        __health__(record)\n"
                "                        with open(Path(args.ckpt_path) / 'paired-steps.jsonl', 'a', encoding='utf-8') as stream:", 1))
    for before, after, count in anchors:
        if source.count(before) != count:
            raise ValueError('Frozen trainer anchor changed: ' + before)
        source = source.replace(before, after)
    return source


def read_records(path, steps, policy):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        i, label = r['step'], r['label_audit']
        if (type(i) is not int or not 0 <= i < steps or r['architecture'] != 'base' or
            r['teacher_policy'] != policy or r['primary_index'] not in (0, 1, 2) or r['auxiliary_index'] is not None):
            raise ValueError('Wrong step/teacher metadata')
        counts = label['query_time_counts']
        if (len(counts) != 60 or sum(counts) != 128 or any(type(x) is not int or x < 0 for x in counts) or
            label['policy'] != policy or label['teacher_type'] != ('online', 'online_cotracker_three', 'offline_cotracker_three')[r['primary_index']]):
            raise ValueError('Wrong 60-frame/128-query teacher labels')
        if r['primary_index'] == 2 and (label['support_used'], label['iterations']) != SETTINGS[policy][1:]:
            raise ValueError('Offline teacher intervention not executed')
        for k in (*LABEL_KEYS, 'support_draw_sha256'):
            if len(label[k]) != 64 or any(c not in '0123456789abcdef' for c in label[k]):
                raise ValueError('Invalid teacher audit digest')
        for k in ('visible_gt09_fraction', 'confidence_mean', 'out_of_bounds_fraction'):
            if not 0 <= label[k] <= 1:
                raise ValueError('Invalid label diagnostic')
        if (not math.isfinite(r['loss']) or not math.isfinite(r['step_seconds']) or r['step_seconds'] <= 0 or
            not math.isfinite(label['teacher_seconds']) or label['teacher_seconds'] < 0):
            raise ValueError('Nonfinite loss/timing')
        health = r['health']
        if any(not math.isfinite(health[k]) for k in ('peak_allocated_mib', 'peak_reserved_mib', 'driver_free_mib')):
            raise ValueError('Invalid memory record')
        if max(health['peak_allocated_mib'], health['peak_reserved_mib']) > 20480 or health['driver_free_mib'] < 2048:
            raise ValueError('GPU memory headroom exceeded')
        signature = lambda x: [x[k] for k in INPUT_KEYS] + [x['label_audit'][k] for k in (*LABEL_KEYS, 'support_draw_sha256', 'query_time_counts')]
        if i in rows and signature(rows[i]) != signature(r):
            raise ValueError('Resumed inputs or labels changed at step ' + str(i))
        rows[i] = r
    if set(rows) != set(range(steps)):
        raise ValueError('Missing audited steps')
    if statistics.median(r['step_seconds'] for i, r in rows.items() if i >= 2) > 8:
        raise ValueError('Training too slow; inspect runtime evidence before continuing')
    return rows


def compare_rows(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError('Pair coverage mismatch')
    changed = 0
    for i, r in candidate.items():
        ref = reference[i]
        if any(ref[k] != r[k] for k in INPUT_KEYS) or any(ref['label_audit'][k] != r['label_audit'][k] for k in ('support_draw_sha256', 'query_time_counts')):
            raise ValueError('Paired training inputs changed at step ' + str(i))
        differs = any(ref['label_audit'][k] != r['label_audit'][k] for k in LABEL_KEYS)
        if r['primary_index'] != 2 and differs:
            raise ValueError('Unmodified teacher labels changed')
        changed += int(r['primary_index'] == 2 and differs)
    if not any(r['primary_index'] == 2 for r in candidate.values()):
        raise ValueError('Offline teacher never sampled')
    return changed
