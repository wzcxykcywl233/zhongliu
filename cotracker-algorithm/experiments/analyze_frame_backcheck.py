"""Label-aware OFFLINE analysis; never used to produce or adjust predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SCORES = ('center', 'center_history', 'center_fixed')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ranks(values):
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2)[inverse]


def statistics(rows, score):
    x = np.array([r[score] for r in rows], dtype=float)
    y = np.array([r['dice'] for r in rows], dtype=float)
    correlation = None
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        correlation = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
    bad = y < .8
    n_bad, n_good = int(bad.sum()), int((~bad).sum())
    auc = None
    if n_bad and n_good:
        auc = float((ranks(-x)[bad].sum() - n_bad*(n_bad+1)/2) / (n_bad*n_good))
    return {'frames': len(rows), 'bad_frames_dice_lt_0_8': n_bad,
            'spearman_score_vs_dice': correlation, 'auc_low_score_detects_bad': auc}


def export_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['no_records']
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def analyze(dataset, results, split):
    import SimpleITK as sitk
    expected = 38 if split == 'test-38' else 10
    cases = sorted(p.name for p in dataset.iterdir() if p.is_dir())
    if len(cases) != expected:
        raise ValueError('Unexpected case count')
    frame_rows, checks, performance = [], [], []
    for iterations in (2, 4, 6):
        reference = f'backcheck_control_i{iterations}'
        candidate = f'backcheck_i{iterations}'
        for profile in (reference, candidate):
            metrics = read_json(results / profile / 'metrics.json')
            if sorted(r['case_id'] for r in metrics['results']) != cases:
                raise ValueError('Metric case set mismatch')
            if profile == reference:
                a = metrics['aggregates']
                performance.append(dict(Split=split, Profile=profile, Iterations=iterations,
                    DSC=a['dice_similarity_coefficient'], HD95=a['hausdorff_distance_95'],
                    MASD=a['surface_distance_average'], CD=a['center_distance'],
                    D98=a['relative_d98_dose'], TimeSec=a['total_time']))
        for case in cases:
            def diagnostic(profile):
                folder = results / profile / 'checkpoint' / 'jobs' / case
                d = read_json(folder / 'diagnostics.json')
                if d['profile'] != profile or d['config']['n_iterations'] != iterations:
                    raise ValueError('Wrong profile or iteration count')
                image = folder / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
                if sha256(image) != d['prediction']['output_file_sha256']:
                    raise ValueError('Prediction file hash mismatch')
                return d, image
            control, _ = diagnostic(reference)
            observed, image = diagnostic(candidate)
            if control['config']['frame_backcheck'] or not observed['config']['frame_backcheck']:
                raise ValueError('Observation flags incorrect')
            equal = control['prediction']['array_sha256'] == observed['prediction']['array_sha256']
            checks.append({'Split': split, 'Case': case, 'Iterations': iterations, 'ExactOutputMatch': equal})
            if not equal:
                export_csv(results / 'backcheck-output-checks.csv', checks)
                raise ValueError(f'Observer changed prediction: {candidate}/{case}')
            # The official evaluator prefers STAPLE labels if present.
            label = dataset / case / 'targets' / f'{case}_staple_labels.mha'
            if not label.exists():
                label = dataset / case / 'targets' / f'{case}_labels.mha'
            truth = sitk.GetArrayFromImage(sitk.ReadImage(str(label))).transpose(2, 0, 1) > 0
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(image))).transpose(2, 0, 1) > 0
            if truth.shape != pred.shape:
                raise ValueError('Prediction/label shape mismatch')
            records = observed['frame_backcheck']
            if [r['frame'] for r in records] != list(range(1, len(truth))):
                raise ValueError('Missing or duplicate frame scores')
            for record in records:
                t = record['frame']
                dice = float(2*(truth[t] & pred[t]).sum() / (truth[t].sum()+pred[t].sum())) if truth[t].any() else None
                enough = record['supported_points'] >= .1 * record['points']
                frame_rows.append({'Split': split, 'Case': case, 'Iterations': iterations, **record,
                                   'dice': dice, 'analysis_eligible': dice is not None and enough
                                   and all(record[s] is not None and np.isfinite(record[s]) for s in SCORES)})
    usable = {n: [r for r in frame_rows if r['Iterations'] == n and r['analysis_eligible']] for n in (2,4,6)}
    common = set.intersection(*({(r['Case'],r['frame']) for r in usable[n]} for n in (2,4,6)))
    summaries, case_stats = [], []
    for n in (2,4,6):
        for scope in ('all_supported', 'common_i2_i4_i6'):
            rows = [r for r in usable[n] if scope == 'all_supported' or (r['Case'],r['frame']) in common]
            for score in SCORES:
                values = []
                for case in cases:
                    stats = statistics([r for r in rows if r['Case'] == case], score)
                    case_stats.append({'Split':split,'Iterations':n,'Scope':scope,'Score':score,'Case':case, **stats})
                    if stats['spearman_score_vs_dice'] is not None:
                        values.append(stats['spearman_score_vs_dice'])
                summaries.append({'Split':split,'Iterations':n,'Scope':scope,'Score':score, **statistics(rows,score),
                                  'macro_case_spearman': float(np.mean(values)) if values else None,
                                  'cases_with_defined_spearman':len(values),
                                  'eligible_frame_fraction': len(rows)/sum(r['Iterations']==n for r in frame_rows)})
    export_csv(results / 'backcheck-frames.csv', frame_rows)
    export_csv(results / 'backcheck-score-summary.csv', summaries)
    export_csv(results / 'backcheck-score-cases.csv', case_stats)
    export_csv(results / 'backcheck-output-checks.csv', checks)
    export_csv(results / f'iteration-performance-{split}.csv', performance)
    summary = {'split':split,'cases':len(cases),'output_pairs':len(checks),'all_outputs_exact':True,
               'scored_frames':{str(n):len(usable[n]) for n in usable}, 'common_frames':len(common),
               'note':'Scores are uncalibrated evidence, not probabilities. Frame Dice is diagnostic only. Official performance comes from metrics.json. No p-values: frames are correlated.'}
    tmp = results / 'backcheck-analysis-summary.json.tmp'
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(results / 'backcheck-analysis-summary.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--split', choices=('test-38','validation-10'), required=True)
    args = parser.parse_args()
    analyze(args.dataset, args.results, args.split)
