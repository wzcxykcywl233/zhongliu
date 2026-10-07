"""Copy official split metrics; report paired branch and frozen reference contrasts."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker'), str(Path(__file__).parent)]
from audit_backbone_growth import file_sha
from summarize_backbone_growth import FIELDS, PROFILE, write_csv
from cotracker.models.core.cotracker.feature_sidecar import KINDS
from cotracker.models.core.cotracker.backbone_growth import save_json


def load_arm(folder, kind, count, parameters):
    path = folder / PROFILE / 'metrics.json'
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    cases = {r['case_id']:r for r in data['results']}
    if len(cases) != count or len(data['results']) != count:
        raise ValueError('wrong case count')
    rms = []
    for case, row in cases.items():
        if not all(math.isfinite(float(row[f])) for f in FIELDS.values()):
            raise ValueError('nonfinite case metric')
        job = folder / PROFILE / 'checkpoint' / 'jobs' / case
        d = json.loads((job / 'diagnostics.json').read_text(encoding='utf-8-sig'))
        if d['profile'] != PROFILE or d['model']['parameters'] != parameters or d['model'].get('feature_sidecar', 'none') != kind:
            raise ValueError('wrong executed feature branch')
        if not d['model'].get('frozen_pretrained'):
            raise ValueError('unmatched backbone computation flags')
        output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
        if not (job / '.complete').is_file() or file_sha(output) != d['prediction']['output_file_sha256']:
            raise ValueError('invalid committed output')
        if kind != 'none':
            if not d['mechanism'].get('sidecar_calls', 0):
                raise ValueError('feature branch not executed')
            rms.append(d['mechanism']['sidecar_residual_rms_max'])
    a = {k:float(data['aggregates'][f]) for k,f in FIELDS.items()}
    a['TimeSec'] = float(data['aggregates']['total_time'])
    if not all(math.isfinite(v) for v in a.values()):
        raise ValueError('nonfinite aggregate')
    return a, cases, file_sha(path), max(rms, default=0.)


def summarize(root, seeds, steps, split):
    count = 38 if split == 'test-38' else 10
    audit = json.loads((root / 'train' / 'training-audit.json').read_text())
    if not audit['complete'] or audit['steps'] != steps or audit['seeds'] != seeds:
        raise ValueError('wrong paired training audit')
    for path, expected in audit['final_checkpoints_sha256'].items():
        if file_sha(root / 'train' / path) != expected:
            raise ValueError('trained checkpoint changed')
    budgets = {(p['seed'], p['kind']):p['parameters'] for p in audit['proofs']}
    folder = root / split
    ref, ref_cases, h, _ = load_arm(folder / 'pretrained', 'none', count, 25385700)
    hashes = {'pretrained/' + PROFILE + '/metrics.json':h}
    rows = [{'Seed':-1,'Kind':'pretrained','Parameters':25385700,'ResidualRmsMax':0., **ref}]
    deltas, case_deltas, collected = [], [], {}
    for seed in seeds:
        arms = {}
        for kind in KINDS:
            path = folder / f'seed_{seed}' / kind
            a, cases, h, rms = load_arm(path, kind, count, budgets[(seed,kind)])
            if set(cases) != set(ref_cases):
                raise ValueError('case IDs differ between arms')
            hashes[str((path / PROFILE / 'metrics.json').relative_to(folder))] = h
            arms[kind] = a, cases
            collected.setdefault(kind, []).append(a)
            rows.append({'Seed':seed,'Kind':kind,'Parameters':budgets[(seed,kind)],'ResidualRmsMax':rms, **a})
        comparisons = [(k, 'pretrained') for k in KINDS] + [('conv','mlp'),('mamba','mlp'),('mamba','conv'),('mamba','mamba_short')]
        for candidate, reference in comparisons:
            a, ac = (ref, ref_cases) if reference == 'pretrained' else arms[reference]
            b, bc = arms[candidate]
            deltas.append({'Seed':seed,'Candidate':candidate,'Reference':reference,
                           **{'Delta_'+k:b[k]-a[k] for k in FIELDS}})
            for case in sorted(ac):
                case_deltas.append({'Seed':seed,'Candidate':candidate,'Reference':reference,'Case':case,
                    **{'Delta_'+k:float(bc[case][f])-float(ac[case][f]) for k,f in FIELDS.items()}})
    summary = []
    for kind, samples in [('pretrained',[ref])] + list(collected.items()):
        row = {'Kind':kind,'N':len(samples),'Parameters':25385700 if kind == 'pretrained' else budgets[(seeds[0],kind)]}
        for k in FIELDS:
            row[k+'_Mean'] = statistics.mean(a[k] for a in samples)
            row[k+'_SD'] = statistics.stdev(a[k] for a in samples) if len(samples) > 1 else 0.
        summary.append(row)
    write_csv(folder / 'feature-sidecar-results.csv', rows)
    write_csv(folder / 'feature-sidecar-summary.csv', summary)
    write_csv(folder / 'feature-sidecar-paired-deltas.csv', deltas)
    write_csv(folder / 'feature-sidecar-paired-cases.csv', case_deltas)
    proof = {'complete':True,'split':split,'cases':count,'seeds':seeds,'training_steps':steps,'metrics_sha256':hashes,
             'note':'Official metrics only for this split. Mamba-style reference scan, not a kernel-speed benchmark. Test-38 already inspected; no blind-test/p-value claim.'}
    save_json(folder / 'feature-sidecar-audit.json', proof)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--split', choices=['validation-10','test-38'], required=True)
    a = p.parse_args()
    summarize(a.root, [int(x) for x in a.seeds.split(',')], a.steps, a.split)
