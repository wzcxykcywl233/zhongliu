"""Official split metrics, architecture checks and paired seed contrasts."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker')]
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.backbone_growth import ARCHITECTURES, PARAMETERS, save_json

PROFILE = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
FIELDS = {'DSC':'dice_similarity_coefficient', 'HD95':'hausdorff_distance_95',
          'MASD':'surface_distance_average', 'CD':'center_distance', 'D98':'relative_d98_dose'}


def write_csv(path, rows):
    import os
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_arm(folder, architecture, count):
    path = folder / PROFILE / 'metrics.json'
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = {row['case_id']: row for row in data['results']}
    if len(rows) != count or len(data['results']) != count:
        raise ValueError('wrong case coverage: ' + str(path))
    for case, row in rows.items():
        if any(not math.isfinite(float(row[field])) for field in FIELDS.values()):
            raise ValueError('non-finite case metric')
        job = folder / PROFILE / 'checkpoint' / 'jobs' / case
        diagnostic = json.loads((job / 'diagnostics.json').read_text(encoding='utf-8-sig'))
        if diagnostic['profile'] != PROFILE or diagnostic['model'] != {'backbone_architecture': architecture, 'parameters': PARAMETERS[architecture]}:
            raise ValueError('executed backbone differs from the intended architecture: ' + str(job))
        output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
        if not (job / '.complete').is_file() or file_sha(output) != diagnostic['prediction']['output_file_sha256']:
            raise ValueError('invalid committed output: ' + str(job))
    aggregates = {key: float(data['aggregates'][field]) for key, field in FIELDS.items()}
    aggregates['TimeSec'] = float(data['aggregates']['total_time'])
    if not all(math.isfinite(value) for value in aggregates.values()):
        raise ValueError('non-finite aggregate metric')
    return aggregates, rows, file_sha(path)


def summarize(root, seeds, steps, split):
    count = 10 if split == 'validation-10' else 38
    folder = root / split
    pretrained, reference_cases, checksum = load_arm(folder / 'pretrained', 'base', count)
    report = json.loads((root / 'train' / 'training-audit.json').read_text(encoding='utf-8'))
    if not report['complete'] or report['seeds'] != seeds or report['steps'] != steps:
        raise ValueError('training audit does not match this evaluation')
    for relative, expected in report['final_checkpoints_sha256'].items():
        if file_sha(root / 'train' / relative) != expected:
            raise ValueError('trained checkpoint changed after audit')
    records = [{'Seed': -1, 'Architecture': 'pretrained', 'Parameters': PARAMETERS['base'], **pretrained}]
    hashes = {'pretrained/' + PROFILE + '/metrics.json': checksum}
    deltas, case_rows, values = [], [], {}
    for seed in seeds:
        arms = {}
        for architecture in ARCHITECTURES:
            path = folder / ('seed_' + str(seed)) / architecture
            aggregate, cases, checksum = load_arm(path, architecture, count)
            if set(cases) != set(reference_cases):
                raise ValueError('case IDs differ between arms')
            arms[architecture] = (aggregate, cases)
            hashes[str((path / PROFILE / 'metrics.json').relative_to(folder))] = checksum
            values.setdefault(architecture, []).append(aggregate)
            records.append({'Seed': seed, 'Architecture': architecture, 'Parameters': PARAMETERS[architecture], **aggregate})
        for candidate, reference in [('time6','base'),('space_time6','base'),('space_time6','time6'),('base','pretrained'),('time6','pretrained'),('space_time6','pretrained')]:
            a, ac = (pretrained, reference_cases) if reference == 'pretrained' else arms[reference]
            b, bc = arms[candidate]
            deltas.append({'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                           **{f'Delta_{key}': b[key] - a[key] for key in FIELDS}})
            for case in sorted(ac):
                case_rows.append({'Seed': seed, 'Candidate': candidate, 'Reference': reference,
                                  'Case': case, 'Cohort': case.split('_')[0],
                                  **{f'Delta_{key}': float(bc[case][field]) - float(ac[case][field]) for key,field in FIELDS.items()}})
    mean_rows = [{'Architecture':'pretrained','N':1,'Parameters':PARAMETERS['base'],
                  **{f'{key}_Mean': pretrained[key] for key in FIELDS},
                  **{f'{key}_SD':0. for key in FIELDS}}]
    for architecture, aggregates in values.items():
        row = {'Architecture':architecture, 'N':len(seeds), 'Parameters':PARAMETERS[architecture]}
        for key in FIELDS:
            samples = [r[key] for r in aggregates]
            row[key + '_Mean'] = statistics.mean(samples)
            row[key + '_SD'] = statistics.stdev(samples) if len(samples) > 1 else 0.
        mean_rows.append(row)
    paired_means = []
    for candidate, reference in dict.fromkeys((row['Candidate'],row['Reference']) for row in deltas):
        subset = [row for row in deltas if (row['Candidate'],row['Reference']) == (candidate,reference)]
        row = {'Candidate':candidate,'Reference':reference,'N':len(subset)}
        for key in FIELDS:
            samples = [value['Delta_' + key] for value in subset]
            row['Delta_' + key + '_Mean'] = statistics.mean(samples)
            row['Delta_' + key + '_SD'] = statistics.stdev(samples) if len(samples) > 1 else 0.
        paired_means.append(row)
    # Match field order across pretrained and trained rows for easy Excel paste.
    fieldnames = list(mean_rows[0])
    mean_rows = [{key:row[key] for key in fieldnames} for row in mean_rows]
    write_csv(folder / 'backbone-growth-results.csv', records)
    write_csv(folder / 'backbone-growth-summary.csv', mean_rows)
    write_csv(folder / 'backbone-growth-paired-deltas.csv', deltas)
    write_csv(folder / 'backbone-growth-paired-summary.csv', paired_means)
    write_csv(folder / 'backbone-growth-paired-cases.csv', case_rows)
    proof = {'complete':True,'split':split,'cases':count,'seeds':seeds,
             'metrics_sha256':hashes,'training_steps':steps,
             'note':'Official metrics from the named split only. Paired seed comparisons separate architecture growth from ordinary fine-tuning. Test-38 is a repeatedly inspected public dataset, not a fresh blind test.'}
    save_json(folder / 'backbone-growth-audit.json', proof)
    print(json.dumps(mean_rows), flush=True)
    print(json.dumps(proof), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--split', choices=['validation-10','test-38'], required=True)
    args = parser.parse_args()
    summarize(args.root, [int(value) for value in args.seeds.split(',')], args.steps, args.split)
