"""Audit matched query/trajectory controls and report official metric contrasts."""
from dataclasses import asdict
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import POINT_DECOUPLING_EXPERIMENTS
from point_decoupling import array_sha, file_sha, load_cache, subset_indices
from diagnose_retune_points import FIELDS, load_metrics

MANIFEST = Path(__file__).with_name('point-decoupling-40-10-38.json')


def atomic_json(path, value):
    pending = path.with_suffix('.json.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def atomic_csv(path, rows):
    pending = path.with_suffix('.csv.tmp')
    with pending.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def audit_case(root, case):
    diagnostics, caches = {}, {}
    for profile, config in POINT_DECOUPLING_EXPERIMENTS.items():
        job = root / profile / 'checkpoint' / 'jobs' / case
        record = json.loads((job / 'diagnostics.json').read_text(encoding='utf-8-sig'))
        output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
        if not (job / '.complete').is_file() or not (job / 'prediction.json').is_file():
            raise ValueError('incomplete checkpoint: ' + profile + '/' + case)
        if record['profile'] != profile or record['config'] != asdict(config):
            raise ValueError('executed config mismatch: ' + profile + '/' + case)
        if file_sha(output) != record['prediction']['output_file_sha256']:
            raise ValueError('output file checksum mismatch: ' + profile + '/' + case)
        diagnostics[profile] = record
        if profile != 'pd_dense_thin':
            metadata, arrays = load_cache(job / 'output' / 'point-cache.npz')
            if metadata['profile'] != profile or metadata['config'] != asdict(config):
                raise ValueError('cached config mismatch')
            if metadata != record['point_decoupling']:
                raise ValueError('logged and cached provenance differs')
            caches[profile] = (metadata, arrays)
    dense_meta, dense = caches['pd_dense']
    for key in ('queries', 'parent_queries', 'trajectories'):
        if not np.array_equal(dense[key], caches['pd_dense_repeat'][1][key]):
            raise ValueError('dense repeat not exact: ' + case + '/' + key)
    if diagnostics['pd_dense']['prediction']['array_sha256'] != diagnostics['pd_dense_repeat']['prediction']['array_sha256']:
        raise ValueError('dense repeat masks differ: ' + case)
    indices = subset_indices(1000, 250).numpy()
    sparse = caches['pd_sparse_matched'][1]
    if not np.array_equal(dense['queries'][:, indices], sparse['queries']):
        raise ValueError('matched sparse queries differ from dense subset: ' + case)
    if not np.array_equal(dense['queries'], sparse['parent_queries']):
        raise ValueError('matched parent queries differ: ' + case)
    for metadata, _ in caches.values():
        for key in ('input_frames_sha256', 'input_target_sha256', 'model_shape', 'native_shape'):
            if metadata[key] != dense_meta[key]:
                raise ValueError('cache input/protocol mismatch: ' + case)
    source = root / 'pd_dense' / 'checkpoint' / 'jobs' / case / 'output' / 'point-cache.npz'
    replay = diagnostics['pd_dense_thin']['point_decoupling']
    if replay['source_profile'] != 'pd_dense' or replay['source_cache_sha256'] != file_sha(source):
        raise ValueError('replay source changed: ' + case)
    if (replay['selected_indices'] != indices.tolist() or
        replay['selected_query_sha256'] != array_sha(dense['queries'][:, indices]) or
        replay['selected_trajectory_sha256'] != array_sha(dense['trajectories'][:, :, indices]) or
        replay['source_trajectory_sha256'] != dense_meta['array_hashes']['trajectories']):
        raise ValueError('reconstruction did not use the fixed dense trajectory: ' + case)
    displacement = np.linalg.norm(sparse['trajectories'][:, 1:] - dense['trajectories'][:, 1:, indices], axis=-1)
    query_shift = np.linalg.norm(sparse['queries'][..., 1:] - caches['pd_sparse_native'][1]['queries'][..., 1:], axis=-1)
    return diagnostics, {
        'Case': case, 'Cohort': case.split('_')[0],
        'MatchedTrajectoryMeanModelPixels': float(displacement.mean()) if displacement.size else 0.,
        'MatchedTrajectoryMaxModelPixels': float(displacement.max()) if displacement.size else 0.,
        'MatchedTrajectoryChangedFraction': float((displacement > 1e-6).mean()) if displacement.size else 0.,
        'NativeQueryShiftMeanModelPixels': float(query_shift.mean()),
        'NativeQueryShiftMaxModelPixels': float(query_shift.max()),
        'DenseReplayUsesExactCache': True,
    }


def analyze(root, split, prior=None):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    count = 10 if split == 'validation-10' else 38
    cases_by_profile = {p: load_metrics(root / p / 'metrics.json', count) for p in manifest['profiles']}
    cases = sorted(cases_by_profile['pd_dense'])
    if any(sorted(rows) != cases for rows in cases_by_profile.values()):
        raise ValueError('profile case IDs do not match')
    for case in cases:
        if any(abs(cases_by_profile['pd_dense'][case][key] - cases_by_profile['pd_dense_repeat'][case][key]) > 1e-8 for key in FIELDS):
            raise ValueError('dense repeat official metrics differ: ' + case)
    official, comparisons, detail, chain = [], [], [], []
    source_hashes = {}
    historical_checks = 0
    historical_sources = {}
    for profile in manifest['profiles']:
        path = root / profile / 'metrics.json'
        source_hashes[str(path.relative_to(root))] = file_sha(path)
        aggregates = json.loads(path.read_text(encoding='utf-8-sig'))['aggregates']
        config = POINT_DECOUPLING_EXPERIMENTS[profile]
        official.append({'Split': split, 'Profile': profile, 'TrackingPoints': config.border_points,
                         'ReconstructionPoints': config.reconstruction_points or config.border_points,
                         **{key: float(aggregates[field]) for key, field in FIELDS.items()},
                         'TimeSec': float(aggregates['total_time']),
                         'ReplayOnly': profile == 'pd_dense_thin'})
    for case in cases:
        diagnostics, probe = audit_case(root, case)
        chain.append(probe)
        if prior is not None:
            for profile, historical in [('pd_dense', 'rt_control'), ('pd_sparse_native', 'rt_points_0')]:
                path = prior / 'final' / split / historical / 'checkpoint' / 'jobs' / case / 'diagnostics.json'
                old = json.loads(path.read_text(encoding='utf-8-sig'))
                old_output = path.parent / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
                if old['profile'] != historical or file_sha(old_output) != old['prediction']['output_file_sha256']:
                    raise ValueError('historical output checkpoint is invalid: ' + historical + '/' + case)
                if old['prediction']['array_sha256'] != diagnostics[profile]['prediction']['array_sha256']:
                    raise ValueError('historical output reproduction failed: ' + profile + '/' + case)
                historical_checks += 1
                historical_sources[str(path.relative_to(prior))] = file_sha(path)
        for pair in manifest['matched_pairs']:
            ref, cand = pair['reference'], pair['candidate']
            a, b = cases_by_profile[ref][case], cases_by_profile[cand][case]
            detail.append({'Split': split, 'Effect': pair['effect'], 'Reference': ref, 'Candidate': cand,
                           'Case': case, 'Cohort': case.split('_')[0],
                           **{f'Delta_{key}': b[key] - a[key] for key in FIELDS},
                           'OutputExact': diagnostics[ref]['prediction']['array_sha256'] == diagnostics[cand]['prediction']['array_sha256']})
    aggregate_map = {r['Profile']: r for r in official}
    for pair in manifest['matched_pairs']:
        a, b = aggregate_map[pair['reference']], aggregate_map[pair['candidate']]
        comparisons.append({'Split': split, **pair, **{f'Delta_{key}': b[key] - a[key] for key in FIELDS}})
    paired_summary = []
    for pair in manifest['matched_pairs']:
        selected = [row for row in detail if row['Effect'] == pair['effect']]
        for cohort in ['all'] + sorted({row['Cohort'] for row in selected}):
            rows = selected if cohort == 'all' else [row for row in selected if row['Cohort'] == cohort]
            for key in FIELDS:
                delta = [row[f'Delta_{key}'] for row in rows]
                sign = 1 if key in ('DSC', 'D98') else -1
                paired_summary.append({'Effect': pair['effect'], 'Cohort': cohort, 'Metric': key,
                                       'Cases': len(rows), 'MeanDelta': sum(delta) / len(delta),
                                       'ImprovedCases': sum(v * sign > 1e-8 for v in delta),
                                       'WorsenedCases': sum(v * sign < -1e-8 for v in delta),
                                       'TiedCases': sum(abs(v) <= 1e-8 for v in delta)})
    prefix = 'point-decoupling-' + split
    atomic_csv(root / (prefix + '-results.csv'), official)
    atomic_csv(root / (prefix + '-matched-deltas.csv'), comparisons)
    atomic_csv(root / (prefix + '-paired-cases.csv'), detail)
    atomic_csv(root / (prefix + '-paired-summary.csv'), paired_summary)
    atomic_csv(root / (prefix + '-trajectory-diagnostics.csv'), chain)
    report = {'complete': True, 'split': split, 'cases': count, 'profiles': manifest['profiles'],
              'exact_dense_repeat': True, 'exact_replay_source': True, 'exact_matched_queries': True,
              'historical_output_checks': historical_checks, 'source_metrics_sha256': source_hashes,
              'historical_diagnostics_sha256': historical_sources,
              'note': 'Official aggregate metrics copied from metrics.json. Pair comparisons diagnose the whole query-density pipeline; they do not isolate transformer attention from memory/occlusion aggregation. Replay time excludes tracking. Test-38 has already been inspected; no blind-test or p-value claim.'}
    atomic_json(root / 'point-decoupling-audit.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--split', choices=['validation-10', 'test-38'], required=True)
    parser.add_argument('--prior', type=Path)
    args = parser.parse_args()
    analyze(args.results, args.split, args.prior)
