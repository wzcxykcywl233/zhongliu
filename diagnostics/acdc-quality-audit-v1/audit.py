"""CPU-only audit of original ACDC and existing external teacher labels. No training."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import io
import json
import os
from pathlib import Path
import re
import traceback

import numpy as np

from qa_core import POLICY, array_sha, label_metrics, mask_metrics, mask_review_flags, nifti_header, read, safe, sha, temporal_metrics
from review import document, raw_previews, table, trajectory_preview

PACKAGE = Path(__file__).resolve().parent


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temp, path)


def save(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_file(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ['note']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    write(path, stream.getvalue())


def freeze(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Inputs/policy/source changed: use a NEW OutputRoot: '+str(path))
    if not path.exists(): save(path, value)


def manifest(audit):
    path = Path(audit) / 'data-manifest.json'
    m = read(path)
    rows = m['patients']
    if (m.get('audit_version') != 2 or not m.get('training_ready') or len(rows) != 100 or
            {r['patient'] for r in rows} != {f'patient{i:03}' for i in range(1, 101)} or
            Counter(r['split'] for r in rows) != Counter({'external-train-80': 80, 'external-dev-20': 20})):
        raise ValueError('Expected frozen native-v2 ACDC 80/20 manifest; no official test data')
    for r in rows:
        if r['cine'] != f"training/{r['patient']}/{r['patient']}_4d.nii.gz":
            raise ValueError('Unexpected cine path')
        for name in r['files_sha256']:
            if Path(name).name != name or '/' in name or '\\' in name or ':' in name:
                raise ValueError('Unsafe manifest filename')
    return m


def load_volume(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(str(path))
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Vector image is unsupported')
    return sitk.GetArrayFromImage(image), {'spacing': list(image.GetSpacing()),
             'origin': list(image.GetOrigin()), 'direction': list(image.GetDirection()), 'dim': image.GetDimension()}


def same_geometry(cine, other):
    if cine['dim'] != 4 or other['dim'] != 3: return False
    return (np.allclose(cine['spacing'][:3], other['spacing'], rtol=1e-5, atol=1e-5) and
            np.allclose(cine['origin'][:3], other['origin'], rtol=1e-5, atol=1e-4) and
            np.allclose(np.array(cine['direction']).reshape(4, 4)[:3, :3],
                        np.array(other['direction']).reshape(3, 3), rtol=1e-5, atol=1e-5))


def patient_audit(data, row, folder):
    folder.mkdir(parents=True, exist_ok=True)
    pid = row['patient']
    source = safe(data, 'training/'+pid)
    files = {name: sha(source / name) if (source / name).is_file() else None for name in row['files_sha256']}
    signature = {'actual_files_sha256': files, 'manifest_row': row}
    freeze(folder / 'inputs.json', signature)
    done = folder / 'receipt.json'
    if done.exists():
        for name, expected in read(done)['outputs'].items():
            if sha(safe(folder, name)) != expected: raise ValueError('QC output changed: '+str(folder))
        return read(folder / 'result.json')
    result = {'patient': pid, 'split': row['split'], 'group': row['group'], 'data_errors': [],
              'review_flags': [], 'expected_annotation_policy': 'ED/ES only; no intermediate dense masks expected',
              'source_verified': files == row['files_sha256'], 'voxel_checks_completed': False}
    if not result['source_verified']: result['data_errors'].append('source_missing_or_changed_since_native_audit')
    try:
        headers = {}
        for name, digest in files.items():
            if name.endswith('.nii.gz') and digest:
                headers[name] = nifti_header(source / name)
                result['review_flags'].extend(headers[name]['flags'])
        result['headers'] = headers
        a, geometry = load_volume(safe(data, row['cine']))
        if list(a.shape) != row['shape_tzyx']:
            raise ValueError('Cine shape changed')
        slices, pairs = temporal_metrics(a)
        result['shape_tzyx'] = list(a.shape)
        result['decoded_cine_sha256'] = array_sha(a.astype(np.float32))
        result['geometry'] = geometry
        result['native_aspect_ratio'] = a.shape[-1] / a.shape[-2]
        result['model_aspect_ratio'] = 512 / 384
        result['aspect_ratio_multiplier'] = (512 / 384) / result['native_aspect_ratio']
        masks, mask_rows = {}, []
        phases = row['annotation_frames_zero_based']
        if phases['ED'] == phases['ES']: result['review_flags'].append('same_ED_ES_index_review')
        for phase in ('ED', 'ES'):
            t = phases[phase]
            if not isinstance(t, int) or not 0 <= t < a.shape[0]: raise ValueError('Invalid ED/ES frame')
            stem = f'{pid}_frame{t+1:02}'
            try:
                ref, rg = load_volume(source / (stem+'.nii.gz'))
                mask, mg = load_volume(source / (stem+'_gt.nii.gz'))
                if ref.shape != a.shape[1:] or mask.shape != a.shape[1:]:
                    raise ValueError('ED/ES array shape mismatch')
                if not same_geometry(geometry, rg) or not same_geometry(geometry, mg):
                    raise ValueError('ED/ES geometry mismatch')
                if not np.allclose(ref, a[t], rtol=1e-5, atol=1e-4):
                    raise ValueError('ED/ES image does not match declared cine frame')
                mask_rows += mask_metrics(mask, phase, geometry['spacing'])
                masks[phase] = mask
                for label in (1, 2, 3):
                    if not (mask == label).any(): result['review_flags'].append(f'{phase}_class_{label}_absent_review')
            except Exception as exc:
                result['data_errors'].append(phase+': '+str(exc))
        if len(masks) == 2:
            result['ED_ES_masks_identical'] = bool(np.array_equal(masks['ED'], masks['ES']))
            result['ED_ES_images_identical'] = bool(np.array_equal(a[phases['ED']], a[phases['ES']]))
            if result['ED_ES_masks_identical']: result['review_flags'].append('identical_ED_ES_masks_review')
            result['ED_ES_label_dice_descriptive_only'] = {
                str(k): (float(2*((masks['ED'] == k) & (masks['ES'] == k)).sum()) /
                         max(1, int((masks['ED'] == k).sum()+(masks['ES'] == k).sum()))) for k in (1, 2, 3)}
        for sr in slices: result['review_flags'].extend(sr['flags'])
        result['review_flags'].extend(mask_review_flags(mask_rows))
        result['slice_flag_counts'] = dict(Counter(f for s in slices for f in s['flags']))
        # Always include a small neutral sample plus flagged slices; all slices have metrics/contact sheets.
        result['preview_slices'] = raw_previews(folder, a, masks, phases, slices)
        result['voxel_checks_completed'] = True
        csv_file(folder / 'slice-metrics.csv', slices)
        csv_file(folder / 'frame-pairs.csv', pairs)
        csv_file(folder / 'mask-slices.csv', mask_rows)
    except Exception as exc:
        result['data_errors'].append(type(exc).__name__+': '+str(exc))
        write(folder / 'error.txt', traceback.format_exc())
    result['review_flags'] = sorted(set(result['review_flags']))
    save(folder / 'result.json', result)
    body = '<p><a href="../../index.html">返回总览</a> | 所有帧号/切片号从 0 开始。</p>'
    body += '<p>红=右心室，绿=心肌，蓝=左心室。仅 ED/ES 帧叠加真实标注；其余帧 NO MASK 是正常情况。GIF末尾回到开头是播放循环，不是新增异常帧。播放速度是展示用，不代表采集帧率。</p>'
    body += '<p>动画左侧保留原像素宽高比，以该切片全序列 min/max 显示；右侧采用 p01/p99 和模型 512×384 宽高比。两侧均缩放到预览尺寸，不是原始像素级显示，也不是物理等比例测量。</p>'
    body += '<pre>'+html.escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre>'
    for name in ('cine.gif', 'ED.png', 'ES.png'):
        if (folder / name).exists(): body += '<h2>'+name+'</h2><img src="'+name+'">'
    write(folder / 'index.html', document(pid+' 质量复核', body))
    outputs = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name not in ('receipt.json', 'inputs.json') and not p.name.endswith('.tmp')}
    save(done, {'complete': True, 'outputs': outputs})
    print('AUDIT '+pid+f" errors={len(result['data_errors'])} review={len(result['review_flags'])}", flush=True)
    return result


def raw_stage(data, m, out):
    rows = [patient_audit(data, r, out / 'raw' / r['patient']) for r in m['patients']]
    duplicates = defaultdict(list)
    for r in rows:
        if 'decoded_cine_sha256' in r:
            duplicates[(tuple(r['shape_tzyx']), r['decoded_cine_sha256'])].append(r['patient'])
    duplicate_groups = [p for p in duplicates.values() if len(p) > 1]
    csv_file(out / 'patients.csv', [{k: r.get(k) for k in ('patient', 'split', 'group', 'source_verified',
             'voxel_checks_completed', 'data_errors', 'review_flags', 'aspect_ratio_multiplier')} for r in rows])
    save(out / 'duplicate-cines.json', {'exact_decoded_duplicates': duplicate_groups,
         'note': 'Exact float32 voxel equality only. No near-duplicate or patient identity certification.'})
    summary = {'complete': True, 'patients_attempted': len(rows),
               'voxel_checks_completed': sum(r['voxel_checks_completed'] for r in rows),
               'patients_with_data_errors': sum(bool(r['data_errors']) for r in rows),
               'patients_with_review_flags': sum(bool(r['review_flags']) for r in rows),
               'patient_review_flag_counts': dict(Counter(f for r in rows for f in r['review_flags'])),
               'exact_duplicate_groups': len(duplicate_groups),
               'split_counts_unchanged': dict(Counter(r['split'] for r in rows)),
               'quality_certified': False, 'automatic_exclusions': 0}
    save(out / 'raw-summary.json', summary)
    return summary


def choose_replays(rows, per_seed):
    """Per-teacher suspicious + systematic controls; selection is explicitly not random."""
    selected = []
    for slot in (0, 1, 2):
        valid = [r for r in rows if r.get('slot') == slot and not r.get('error')]
        if not valid: continue
        ranked = sorted(valid, key=lambda r: (-r['jump_pair_fraction'], -r['query_anchor_error_max'], r['step']))
        selected += ranked[:2]
        selected += [valid[0], valid[len(valid)//2]]
    unique = {r['step']: r for r in selected}
    return sorted(unique.values(), key=lambda r: r['step'])[:per_seed]


def validate_cached_label(source, step_row, manifest_hash, patients, pool=None):
    import torch
    step = step_row['step']
    path = source / 'labels' / f'{step:04}.pth'
    receipt = read(path.with_suffix('.json'))
    digest = sha(path)
    if (receipt.get('complete') is not True or receipt['manifest_sha256'] != manifest_hash or
            receipt['files_sha256'].get(f'full/labels/{step:04}.pth') != digest or
            step_row['label_sha256'] != digest):
        raise ValueError('Label file/proof differs from the consumed training record')
    label = torch.load(path, map_location='cpu', weights_only=True)
    sig = label['signature']
    if (sig != receipt['signature'] or sig['step'] != step or sig['slot'] != step_row['slot'] or
            sig['patient'] != step_row['patient'] or sig['slot'] not in (0, 1, 2)):
        raise ValueError('Cached label signature differs from training step')
    if sig['patient'] not in patients or patients[sig['patient']]['split'] != 'external-train-80':
        raise ValueError('Training labels reference non-training patient')
    if pool is not None and label.get('audit', {}).get('teacher_setting') != pool[sig['slot']]:
        raise ValueError('Recorded teacher configuration differs from study manifest')
    mr = patients[sig['patient']]
    sampling = sig['sampling']
    if sampling['source_shape_tzyx'] != mr['shape_tzyx']:
        raise ValueError('Sampling shape differs from manifest')
    frames, z = sampling['frames'], sampling['slice_z']
    if (not isinstance(z, int) or not 0 <= z < mr['shape_tzyx'][1] or not frames or
            frames != list(range(frames[0], frames[0]+len(frames))) or
            frames[0] < 0 or frames[-1] >= mr['shape_tzyx'][0] or
            len(frames) != min(16, mr['shape_tzyx'][0])):
        raise ValueError('Invalid native sampling provenance')
    stats = label_metrics(*(label[k].numpy() for k in ('tracks', 'visibility', 'confidence', 'queries')))
    if stats['frames'] != len(frames) or stats['queries'] != 128:
        raise ValueError('Label shape/budget mismatch')
    return {'patient': sig['patient'], 'slot': sig['slot'], 'slice_z': z,
            'native_start': frames[0], 'label_sha256': digest, **stats}


def replay(data, mr, label, folder, step):
    """Reconstruct exact historical CPU input and verify its recorded tensor hash."""
    import torch
    import torch.nn.functional as F
    sig = label['signature']
    sampling = sig['sampling']
    a, _ = load_volume(safe(data, mr['cine']))
    if sha(safe(data, mr['cine'])) != mr['files_sha256'][Path(mr['cine']).name]:
        raise ValueError('Source changed since training')
    z, frames = sampling['slice_z'], sampling['frames']
    if not isinstance(z, int) or not 0 <= z < a.shape[1]: raise ValueError('Replay slice out of range')
    if (not frames or frames != list(range(frames[0], frames[0]+len(frames))) or
            frames[0] < 0 or frames[-1] >= len(a) or len(frames) != min(16, len(a))):
        raise ValueError('Replay must use native contiguous frames')
    plane = a[:, z].astype(np.float32)
    low, high = (float(x) for x in np.percentile(plane, [1, 99]))
    if high <= low: low, high = float(plane.min()), float(plane.max())
    if high <= low or low != sampling['intensity_low'] or high != sampling['intensity_high']:
        raise ValueError('Replay normalization mismatch')
    v = plane[frames].copy()
    v -= np.float32(low)
    v /= np.float32(high-low)
    np.clip(v, 0, 1, out=v)
    v *= np.float32(255)
    video = F.interpolate(torch.from_numpy(v)[:, None], size=(384, 512), mode='bilinear', align_corners=False)
    video = video.clamp(0, 255).expand(-1, 3, -1, -1).contiguous()
    if array_sha(video.numpy()) != sig['video_sha256']:
        raise ValueError('Exact input replay hash differs; do not interpret overlays as historical input')
    if label['tracks'].shape[1] != len(video): raise ValueError('Replay/track length mismatch')
    trajectory_preview(folder / f'{step:04}.gif', video.numpy(), label['tracks'].numpy(), label['visibility'].numpy())
    # A fixed image location changing intensity is only review evidence, not correspondence truth.
    p = label['tracks'].numpy()[0]
    q = label['queries'].numpy()[0]
    static = np.linalg.norm(np.ptp(p, axis=0), axis=-1) <= .1
    moving_intensity = []
    for n in np.where(static)[0]:
        x, y = np.round(q[n, 1:]).astype(int)
        patch = video[:, 0, max(0, y-2):min(384, y+3), max(0, x-2):min(512, x+3)].numpy()
        if patch.size and np.abs(np.diff(patch, axis=0)).mean() > 1.:
            moving_intensity.append(int(n))
    return {'input_replay_exact': True, 'slice_z': z, 'native_frames': frames,
            'static_tracks_in_changing_intensity_review': moving_intensity,
            'gif': f'{step:04}.gif', 'note': 'Intensity change may be noise/contrast change; not proof of failed tracking.'}


def label_stage(data, cache, m, out, replay_count):
    import torch
    torch.set_num_threads(2)
    patients = {r['patient']: r for r in m['patients']}
    all_rows, replay_rows, pools = [], [], {}
    expected = 0
    for seed in range(3):
        source = cache / f'seed_{seed}'
        plan = read(source / 'manifest.json')
        if (plan['external_manifest_sha256'] != sha(Path(m['_path'])) or plan['rule']['external_steps'] != 3000 or
                plan['rule']['seed'] != seed or len(plan['teacher_pool']) != 3):
            raise ValueError('Expected matching full-v1 external study with 3000 labels per seed')
        pools[seed] = plan['teacher_pool']
        steps = read(source / 'steps.json')
        proof = read(source / 'verified.json')
        manifest_hash = sha(source / 'manifest.json')
        if (proof.get('complete') is not True or proof['manifest_sha256'] != manifest_hash or
                proof['files_sha256'].get('full/external/B1/steps.json') != sha(source / 'steps.json') or
                len(steps) != 3000 or [r['step'] for r in steps] != list(range(3000))):
            raise ValueError('Missing/stale completed external-step proof')
        folder = out / 'labels' / f'seed_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        freeze(folder / 'inputs.json', {'manifest_sha256': manifest_hash, 'steps_sha256': sha(source/'steps.json'),
                                      'verified_sha256': sha(source/'verified.json')})
        rows = []
        for st in steps:
            step = st['step']
            expected += 1
            row = {'seed': seed, 'step': step, 'error': ''}
            try:
                row.update(validate_cached_label(source, st, manifest_hash, patients, pools[seed]))
            except Exception as exc:
                row['error'] = type(exc).__name__+': '+str(exc)
            rows.append(row)
            if (step+1) % 250 == 0: print(f'LABEL seed={seed} {step+1}/3000', flush=True)
        csv_file(folder / 'metrics.csv', rows)
        for row in choose_replays(rows, replay_count):
            step = row['step']
            rr = {'seed': seed, 'step': step, 'patient': row['patient'], 'slot': row['slot'], 'error': ''}
            try:
                label = torch.load(source / 'labels' / f'{step:04}.pth', map_location='cpu', weights_only=True)
                rr.update(replay(data, patients[row['patient']], label, folder, step))
            except Exception as exc: rr['error'] = type(exc).__name__+': '+str(exc)
            replay_rows.append(rr)
        all_rows += rows
    csv_file(out / 'label-metrics.csv', all_rows)
    csv_file(out / 'label-replays.csv', replay_rows)
    groups = []
    for seed in range(3):
        for slot in (0, 1, 2):
            group = [r for r in all_rows if r['seed'] == seed and r.get('slot') == slot and not r['error']]
            g = {'seed': seed, 'teacher_slot': slot, 'valid_steps': len(group), **pools[seed][slot]}
            for k in ('visible_fraction', 'confidence_mean', 'out_of_bounds_fraction', 'static_track_fraction', 'jump_pair_fraction'):
                g[k+'_mean'] = float(np.mean([r[k] for r in group])) if group else None
            groups.append(g)
    csv_file(out / 'teacher-summary.csv', groups)
    sampled = defaultdict(list)
    for row in all_rows:
        if not row['error']: sampled[(row['seed'], row['patient'], row['slice_z'], row['slot'])].append(row)
    sampling_rows = []
    for (seed, patient, z, slot), group in sorted(sampled.items()):
        sampling_rows.append({'seed': seed, 'patient': patient, 'slice_z': z, 'teacher_slot': slot,
                              'sampled_steps': len(group),
                              'static_track_fraction_mean': float(np.mean([r['static_track_fraction'] for r in group])),
                              'jump_pair_fraction_mean': float(np.mean([r['jump_pair_fraction'] for r in group]))})
    csv_file(out / 'sampling-slices.csv', sampling_rows)
    summary = {'complete': True, 'labels_expected': expected, 'labels_attempted': len(all_rows),
               'invalid_labels_or_proofs': sum(bool(r['error']) for r in all_rows),
               'replays_attempted': len(replay_rows), 'replay_failures': sum(bool(r['error']) for r in replay_rows),
               'replay_sampling': 'Up to 12/seed: two highest jump/anchor-risk + first/middle per teacher, deduplicated. Not prevalence sample.',
               'quality_certified': False, 'automatic_exclusions': 0}
    save(out / 'labels-summary.json', summary)
    save(out / 'replays.json', replay_rows)
    return summary


def report(out):
    summary = {'complete': True, 'raw': read(out/'raw-summary.json') if (out/'raw-summary.json').exists() else None,
               'cached_labels': read(out/'labels-summary.json') if (out/'labels-summary.json').exists() else None,
               'quality_certified': False, 'training_started': False, 'automatic_exclusions': 0,
               'note': 'Complete means requested checks finished, NOT dataset/label quality certified. No test data or student metrics used.'}
    body = '<p>只读质量检查。标记用于人工复核，不等于坏数据；没有删除、改写、自动筛选或训练。帧号和切片号从 0 开始。</p>'
    body += '<p>ACDC 只有 ED/ES 人工分割，不应期待逐帧掩膜。教师伪轨迹不是真值。静止背景、病理性弱运动、正常心脏形变和播放循环均可能被误读。</p>'
    body += '<pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>固定提示阈值（不是排除标准）</h2><pre>'+html.escape(json.dumps(POLICY, ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>病例复核</h2><ul>'
    for path in sorted((out/'raw').glob('patient*/result.json')):
        r = read(path)
        body += f'<li><a href="raw/{r["patient"]}/index.html">{r["patient"]}</a> {html.escape(r["split"])}; '
        body += html.escape(str(r['data_errors']+r['review_flags']))+'</li>'
    body += '</ul><h2>教师轨迹：经输入哈希复核的抽样回放</h2>'
    if (out/'replays.json').exists():
        for r in read(out/'replays.json'):
            body += '<p>'+html.escape(str(r))+'</p>'
            if r.get('input_replay_exact') and not r['error']:
                body += f'<details><summary>打开回放 seed {r["seed"]} step {r["step"]}</summary><img loading="lazy" src="labels/seed_{r["seed"]}/{r["gif"]}"></details>'
    body += '<p>缺失标注语义、细微伪影、近重复病例身份和真实对应点误差不能由本脚本自动认证。保留困难病例，按切片/片段/轨迹复核；未作病种重平衡或新数据划分。</p>'
    write(out/'index.html', document('ACDC / MedCoTracker 质量复核', body))
    save(out / 'summary.json', summary)
    print('COMPLETE '+str(out/'summary.json'), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('raw', 'labels', 'all'))
    p.add_argument('--data', type=Path, default=Path('/data'))
    p.add_argument('--audit', type=Path, default=Path('/audit'))
    p.add_argument('--cache', type=Path, default=Path('/cache'))
    p.add_argument('--output', type=Path, default=Path('/output'))
    p.add_argument('--runtime-id', required=True)
    p.add_argument('--replays-per-seed', type=int, choices=range(0, 13), default=12)
    a = p.parse_args()
    for source in (a.data, a.audit, a.cache, PACKAGE):
        s, o = source.resolve(), a.output.resolve()
        if o == s or s in o.parents or o in s.parents: raise ValueError('Output must not overlap inputs')
    a.output.mkdir(parents=True, exist_ok=True)
    m = manifest(a.audit)
    m['_path'] = str(a.audit/'data-manifest.json')
    freeze(a.output/'run-config.json', {'schema': 1, 'policy': POLICY, 'manifest_sha256': sha(m['_path']),
           'source_sha256': {x.name: sha(x) for x in sorted(PACKAGE.iterdir()) if x.suffix in ('.py', '.ps1')},
           'runtime_id': a.runtime_id, 'replays_per_seed': a.replays_per_seed})
    save(a.output/'summary.json', {'complete': False, 'stage': a.stage, 'note': 'Audit in progress or interrupted; no quality certification.'})
    if a.stage in ('raw', 'all'): raw_stage(a.data, m, a.output)
    if a.stage in ('labels', 'all'): label_stage(a.data, a.cache, m, a.output, a.replays_per_seed)
    report(a.output)


if __name__ == '__main__': main()
