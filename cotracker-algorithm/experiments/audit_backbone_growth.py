"""Verify full paired training coverage and checkpoint architectures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker')]
from cotracker.models.core.cotracker.backbone_growth import ARCHITECTURES, PARAMETERS, grow_backbone, save_json
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def records(path, steps, architecture):
    rows = {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        row = json.loads(line)
        index = row['step']
        if row['architecture'] != architecture or not isinstance(index, int) or not 0 <= index < steps:
            raise ValueError('invalid audit record: ' + str(path) + ':' + str(number))
        if row['auxiliary_index'] is not None:
            raise ValueError('unexpected auxiliary teacher')
        signature = tuple(json.dumps(row[key], sort_keys=True) for key in
                          ('case_names','queries_sha256','video_sha256','primary_index','auxiliary_index'))
        if index in rows and signature != rows[index]:
            raise ValueError('resumed step input changed: ' + str(index))
        rows[index] = signature
    if set(rows) != set(range(steps)):
        raise ValueError('missing paired audit steps: ' + str(path))
    return rows


def audit(root, seeds, steps):
    proofs, final_hashes = [], {}
    torch.set_num_threads(1)
    for seed in seeds:
        reference, config_reference = None, None
        for architecture in ARCHITECTURES:
            folder = root / ('seed_' + str(seed)) / architecture
            init = json.loads((folder / 'initialization-report.json').read_text(encoding='utf-8'))
            if init['architecture'] != architecture or init['parameters'] != PARAMETERS[architecture] or not init['initialization_outputs_exact'] or any(init['initialization_max_abs_difference']):
                raise ValueError('initialization proof failed: ' + str(folder))
            config = json.loads((folder / 'run-config.json').read_text(encoding='utf-8-sig'))
            config = {key:value for key,value in config.items() if key not in ('architecture','profile')}
            if config_reference is None:
                config_reference = config
            elif config != config_reference:
                raise ValueError('training protocol differs between architectures')
            rows = records(folder / 'paired-steps.jsonl', steps, architecture)
            if reference is None:
                reference = rows
            elif rows != reference:
                raise ValueError('teacher/case/query/video sequence differs: ' + str(folder))
            final = folder / 'cotracker_three_final.pth'
            checkpoint = torch.load(final, map_location='cpu', weights_only=True)
            if checkpoint['backbone_architecture'] != architecture or checkpoint['total_steps'] != steps:
                raise ValueError('wrong final architecture/step: ' + str(final))
            model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            grow_backbone(model, architecture)
            model.load_state_dict(checkpoint['model'], strict=True)
            count = sum(parameter.numel() for parameter in model.parameters())
            added_outputs = [parameter.detach() for name, parameter in model.named_parameters()
                             if (any(f'{group}.{i}.' in name for group in
                                     ('time_blocks','space_virtual_blocks','space_virtual2point_blocks','space_point2virtual_blocks') for i in (3,4,5)))
                             and ('to_out.weight' in name or 'mlp.fc2.weight' in name)]
            changed = sum(int(bool(torch.count_nonzero(parameter))) for parameter in added_outputs)
            if architecture != 'base' and changed == 0:
                raise ValueError('added layers did not learn a residual update')
            final_hashes[str(final.relative_to(root))] = file_sha(final)
            proofs.append({'seed': seed, 'architecture': architecture, 'paired_steps': len(rows),
                           'parameters': count, 'added_output_projections_nonzero': changed})
            del model, checkpoint
    report = {'complete': True, 'seeds': seeds, 'steps': steps, 'proofs': proofs,
              'final_checkpoints_sha256': final_hashes,
              'paired_teacher_case_query_video_sequences': True}
    save_json(root / 'training-audit.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    audit(args.root, [int(value) for value in args.seeds.split(',')], args.steps)
