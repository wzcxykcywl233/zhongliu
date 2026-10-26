"""Read-only ACDC QC primitives. Thresholds flag review, never certify bad anatomy."""
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

VERSION = 1
POLICY = {
    'version': VERSION,
    'temporal_jump_mad_multiplier': 8.,
    'temporal_jump_normalized_floor': .10,
    'static_track_max_extent_pixels': .1,
    'speed_fraction_of_input_diagonal': .05,
    'acceleration_fraction_of_input_diagonal': .08,
    'query_anchor_error_pixels': 1.,
    'visibility_threshold': .9,
    'input_hw': [384, 512],
    'interpretation': 'Heuristic review flags, not exclusion rules or correspondence ground truth.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe relative path')
    parts = relative.split('/')
    if any(x in ('', '.', '..') for x in parts):
        raise ValueError('Unsafe relative path')
    result = (Path(root) / relative).resolve()
    result.relative_to(Path(root).resolve())
    return result


def nifti_header(path):
    """Inspect NIfTI-1 header independently of ITK's geometry interpretation."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        b = stream.read(348)
    if len(b) != 348:
        raise ValueError('Truncated NIfTI header')
    endian = next((e for e in ('<', '>') if struct.unpack(e + 'i', b[:4])[0] == 348), None)
    if endian is None or b[344:347] != b'n+1':
        raise ValueError('Expected single-file NIfTI-1; unsupported header needs manual review')
    def unpack(fmt, offset):
        return struct.unpack_from(endian + fmt, b, offset)
    dim = unpack('8h', 40)
    pix = unpack('8f', 76)
    qcode, scode = unpack('2h', 252)
    s = np.eye(4)
    s[:3] = np.array(unpack('12f', 280)).reshape(3, 4)
    qb, qc, qd = unpack('3f', 256)
    v = np.array([qb, qc, qd], dtype=float)
    squared = float(v @ v)
    if squared > 1.:
        v /= math.sqrt(squared)
        a = 0.
    else:
        a = math.sqrt(1. - squared)
    qb, qc, qd = v
    rotation = np.array([
        [a*a+qb*qb-qc*qc-qd*qd, 2*(qb*qc-a*qd), 2*(qb*qd+a*qc)],
        [2*(qb*qc+a*qd), a*a+qc*qc-qb*qb-qd*qd, 2*(qc*qd-a*qb)],
        [2*(qb*qd-a*qc), 2*(qc*qd+a*qb), a*a+qd*qd-qc*qc-qb*qb]])
    q = np.eye(4)
    q[:3, :3] = rotation @ np.diag([pix[1], pix[2], pix[3] * (-1 if pix[0] < 0 else 1)])
    q[:3, 3] = unpack('3f', 268)
    finite = np.isfinite(pix).all() and np.isfinite(s).all() and np.isfinite(q).all()
    if not finite:
        raise ValueError('Nonfinite NIfTI geometry')
    flags = []
    if qcode and scode and not np.allclose(q, s, rtol=1e-4, atol=1e-3):
        flags.append('qform_sform_disagree_review')
    if scode and not np.allclose(np.linalg.norm(s[:3, :3], axis=0), pix[1:4], rtol=1e-3, atol=1e-4):
        flags.append('sform_scale_disagrees_with_pixdim_review')
    if not qcode and not scode:
        flags.append('no_coded_affine_review')
    return {'shape_xyzt': list(dim[1:dim[0]+1]), 'pixdim': list(pix), 'qform_code': qcode,
            'sform_code': scode, 'qform_ras': q.tolist(), 'sform_ras': s.tolist(),
            'xyzt_units': b[123], 'flags': flags}


def temporal_metrics(array):
    """Every slice and every native adjacent frame; not a motion estimator."""
    a = np.asarray(array)
    if a.ndim != 4 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite T,Z,Y,X cine with >=2 frames')
    slices, pairs = [], []
    for z in range(a.shape[1]):
        p = a[:, z].astype(np.float32)
        low, high = np.percentile(p, [1, 99])
        scale = float(high-low)
        normalized = np.clip((p-low) / max(scale, 1e-12), 0, 1)
        d = np.abs(np.diff(normalized, axis=0)).mean(axis=(1, 2))
        median = float(np.median(d))
        mad = float(np.median(np.abs(d-median)))
        threshold = max(POLICY['temporal_jump_normalized_floor'], median + 8 * 1.4826 * mad)
        hashes = [array_sha(frame) for frame in p]
        duplicates = [t for t in range(1, len(p)) if hashes[t] == hashes[t-1]]
        jumps = [t+1 for t, value in enumerate(d) if value > threshold]
        flags = []
        if float(np.ptp(p)) == 0: flags.append('constant_intensity_slice')
        if len(set(hashes)) == 1: flags.append('temporally_static_slice_review')
        elif duplicates: flags.append('adjacent_duplicate_frames_review')
        if jumps: flags.append('intensity_jump_review')
        slices.append({'slice_z': z, 'frames': len(p), 'p01': float(low), 'p99': float(high),
                       'unique_frames': len(set(hashes)), 'adjacent_duplicate_frames': duplicates,
                       'jump_frames': jumps, 'mean_temporal_change': float(d.mean()),
                       'clip_low_fraction': float((p < low).mean()),
                       'clip_high_fraction': float((p > high).mean()), 'flags': flags})
        for t, value in enumerate(d):
            pairs.append({'slice_z': z, 'from_frame': t, 'to_frame': t+1,
                          'normalized_mad': float(value), 'exact_duplicate': hashes[t] == hashes[t+1],
                          'jump_threshold': threshold, 'review_jump': t+1 in jumps})
    return slices, pairs


def mask_metrics(mask, phase, spacing):
    a = np.asarray(mask)
    if a.ndim != 3 or not np.isfinite(a).all() or not set(np.unique(a)).issubset({0, 1, 2, 3}):
        raise ValueError('Invalid mask dimensions, finite values or label IDs')
    rows = []
    for label in (1, 2, 3):
        region = a == label
        counts = region.sum(axis=(1, 2))
        for z, count in enumerate(counts):
            coords = np.argwhere(region[z])
            center = coords.mean(axis=0) if count else [None, None]
            rows.append({'phase': phase, 'label': label, 'slice_z': z, 'pixels': int(count),
                         'area_mm2': float(count * spacing[0] * spacing[1]),
                         'centroid_y': None if not count else float(center[0]),
                         'centroid_x': None if not count else float(center[1])})
    return rows


def mask_review_flags(rows):
    """Slice-to-slice shape flags, not temporal jumps or medical diagnoses."""
    flags = []
    for phase in ('ED', 'ES'):
        for label in (1, 2, 3):
            group = sorted((r for r in rows if r['phase'] == phase and r['label'] == label), key=lambda r: r['slice_z'])
            occupied = [i for i, r in enumerate(group) if r['pixels'] > 0]
            if occupied:
                for i in range(occupied[0], occupied[-1]+1):
                    if group[i]['pixels'] == 0:
                        flags.append(f'{phase}_class_{label}_internal_empty_slice_{group[i]["slice_z"]}_review')
            for left, right in zip(group, group[1:]):
                if min(left['pixels'], right['pixels']) < 16: continue
                ratio = max(left['pixels'], right['pixels']) / min(left['pixels'], right['pixels'])
                shift = math.hypot(left['centroid_x']-right['centroid_x'], left['centroid_y']-right['centroid_y'])
                if ratio > 4 or shift > 2 * math.sqrt(max(left['pixels'], right['pixels'])):
                    flags.append(f'{phase}_class_{label}_slice_transition_{left["slice_z"]}_{right["slice_z"]}_review')
    return flags


def label_metrics(tracks, visibility, confidence, queries, height=384, width=512):
    p, v, c, q = [np.asarray(x) for x in (tracks, visibility, confidence, queries)]
    if p.ndim != 4 or p.shape[0] != 1 or p.shape[-1] != 2 or p.shape[1] < 2:
        raise ValueError('Expected pseudo-tracks [1,T,N,2]')
    t, n = p.shape[1:3]
    if v.shape != (1, t, n) or c.shape != v.shape or q.shape != (1, n, 3):
        raise ValueError('Pseudo-label tensor shapes disagree')
    if not all(np.isfinite(x).all() for x in (p, v, c, q)):
        raise ValueError('Nonfinite pseudo-labels')
    if np.any((v < 0) | (v > 1)) or np.any((c < 0) | (c > 1)):
        raise ValueError('V/C outside [0,1]')
    times = q[0, :, 0]
    if np.any(times != times.astype(int)) or np.any((times < 0) | (times >= t)):
        raise ValueError('Invalid query time')
    # Match upstream torch.rand * [W,H] and label_audit's half-open extent.
    if np.any((q[0, :, 1:] < 0) | (q[0, :, 1:] >= [width, height])):
        raise ValueError('Query outside input')
    p, v, c = p[0], v[0], c[0]
    visible = v > POLICY['visibility_threshold']
    speed = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    acc = np.linalg.norm(np.diff(p, n=2, axis=0), axis=-1)
    diagonal = math.hypot(height, width)
    jump = speed > POLICY['speed_fraction_of_input_diagonal'] * diagonal
    jerk = acc > POLICY['acceleration_fraction_of_input_diagonal'] * diagonal
    pair_visible = visible[1:] & visible[:-1]
    extent = np.linalg.norm(np.ptp(p, axis=0), axis=-1)
    anchor = np.linalg.norm(p[times.astype(int), np.arange(n)] - q[0, :, 1:], axis=-1)
    oob = ((p < 0) | (p >= [width, height])).any(axis=-1)
    flags = []
    if jump.any(): flags.append('track_jump_review')
    if jerk.any(): flags.append('track_acceleration_review')
    if (extent <= .1).any(): flags.append('static_track_review_not_bad_by_itself')
    if (anchor > 1).any(): flags.append('query_anchor_error_review')
    if oob.any(): flags.append('out_of_bounds_review')
    if not visible.any(): flags.append('no_visible_pseudo_labels')
    return {'frames': t, 'queries': n, 'visible_fraction': float(visible.mean()),
            'confidence_mean': float(c.mean()), 'out_of_bounds_fraction': float(oob.mean()),
            'static_track_fraction': float((extent <= .1).mean()),
            'jump_pair_fraction': float(jump.mean()),
            'visible_jump_count': int((jump & pair_visible).sum()),
            'visible_pair_count': int(pair_visible.sum()),
            'acceleration_fraction': float(jerk.mean()) if jerk.size else 0.,
            'query_anchor_error_max': float(anchor.max()), 'speed_max_pixels': float(speed.max()),
            'first_jump_frame': int(np.argwhere(jump)[0, 0]+1) if jump.any() else None,
            'flags': flags}
