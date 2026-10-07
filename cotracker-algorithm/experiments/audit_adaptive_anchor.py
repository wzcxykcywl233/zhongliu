"""Verify identity diagnostic arm, output files, rollback activation and bounds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from audit_backbone_growth import file_sha


def audit(root, split):
    expected = 38 if split == 'test-38' else 10
    profiles = ['anchor_fixed', 'anchor_diagnostic', 'anchor_reference', 'anchor_endpoint',
                'anchor_rollback', 'anchor_rollback_sensitive', 'anchor_rollback_long']
    reference, hashes, report = None, {}, []
    for profile in profiles:
        metrics_path = root / profile / 'metrics.json'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
        ids = {row['case_id'] for row in metrics['results']}
        if len(ids) != expected or len(metrics['results']) != expected or (reference is not None and ids != set(reference)):
            raise ValueError('wrong case coverage')
        arrays, total, drops, fallback = {}, 0, 0, 0
        for case in sorted(ids):
            job = root / profile / 'checkpoint' / 'jobs' / case
            d = json.loads((job / 'diagnostics.json').read_text(encoding='utf-8-sig'))
            output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
            if not (job / '.complete').is_file() or d['profile'] != profile or file_sha(output) != d['prediction']['output_file_sha256']:
                raise ValueError('invalid committed prediction')
            arrays[case] = d['prediction']['array_sha256']
            mech = d['mechanism']
            if profile != 'anchor_reference':
                if mech.get('anchor_decisions', 0) <= 0 and mech['input_frames'] > 1:
                    raise ValueError('anchor logic never evaluated')
                if mech.get('anchor_clip_frames_max', 0) > d['config']['anchor_max_age'] + 1:
                    raise ValueError('anchor clip exceeded bound')
                total += mech.get('anchor_decisions', 0)
                drops += mech.get('anchor_sustained_drops', 0)
                fallback += mech.get('anchor_global_fallback_points', 0)
        if reference is None:
            reference = arrays
        if profile == 'anchor_diagnostic' and arrays != reference:
            raise ValueError('diagnostic-only mode changed outputs')
        hashes[profile + '/metrics.json'] = file_sha(metrics_path)
        report.append({'profile': profile, 'decisions': total, 'sustained_drops': drops,
                       'drop_fraction': drops / max(total, 1), 'fallback_points': fallback,
                       'output_cases_changed_vs_fixed': sum(arrays[c] != reference[c] for c in arrays)})
    proof = {'complete': True, 'split': split, 'cases': expected, 'diagnostic_outputs_exact': True,
             'profiles': report, 'metrics_sha256': hashes,
             'note': 'No training. Relative/absolute raw-local V/C evidence is uncalibrated. Test-38 is already inspected, not blind.'}
    path = root / 'anchor-audit.json'
    path.with_suffix('.tmp').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', choices=['validation-10', 'test-38'], required=True)
    a = p.parse_args()
    audit(a.root, a.split)
