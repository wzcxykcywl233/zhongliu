"""Strict paired-input audit and byte-identical frozen backbone proof."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker'), str(Path(__file__).parent)]
from audit_backbone_growth import records, file_sha
from cotracker.models.core.cotracker.feature_sidecar import KINDS, attach_sidecar, tensor_hash
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.backbone_growth import save_json


def audit(root, seeds, steps, output=None):
    proofs, hashes = [], {}
    torch.set_num_threads(1)
    for seed in seeds:
        reference, reference_config, base_sha, short_initial = None, None, None, None
        for kind in KINDS:
            folder = root / f'seed_{seed}' / kind
            init = json.loads((folder / 'initialization-report.json').read_text())
            info = init['sidecar']
            if info['kind'] != kind or not info['zero_projection_verified'] or not init['initialization_outputs_exact']:
                raise ValueError('initialization failed: ' + str(folder))
            if not info['initialization_outputs_exact'] or any(info['initialization_max_abs_difference']):
                raise ValueError('actual sidecar initialization outputs changed')
            if info.get('initialization_reference') != 'frozen-pretrained' or not info.get('freeze_transition_is_diagnostic'):
                raise ValueError('initialization reference is not the matched frozen control')
            if base_sha is None:
                base_sha = info['base_tensor_sha256']
            if base_sha != info['base_tensor_sha256']:
                raise ValueError('starting base differs between arms')
            if kind == 'mamba_short':
                short_initial = info['initial_sidecar_sha256']
            if kind == 'mamba' and info['initial_sidecar_sha256'] != short_initial:
                raise ValueError('long/short Mamba starting branch weights differ')
            rows = records(folder / 'paired-steps.jsonl', steps, 'base')
            config = json.loads((folder / 'run-config.json').read_text(encoding='utf-8-sig'))
            common = {k:v for k,v in config.items() if k not in ('kind', 'profile')}
            if reference is None:
                reference, reference_config = rows, common
            if rows != reference or common != reference_config:
                raise ValueError('paired inputs/protocol differ: ' + str(folder))
            final = folder / 'cotracker_three_final.pth'
            ckpt = torch.load(final, map_location='cpu', weights_only=True)
            if ckpt['total_steps'] != steps or ckpt['feature_sidecar'] != kind or ckpt['backbone_architecture'] != 'base':
                raise ValueError('wrong final checkpoint metadata')
            model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            current = attach_sidecar(model, kind)
            model.load_state_dict(ckpt['model'], strict=True)
            if tensor_hash(model.state_dict()) != base_sha:
                raise ValueError('frozen backbone changed: ' + str(final))
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError('nonfinite final parameters')
            if not bool(model.updateformer.feature_sidecar.output.weight.count_nonzero()):
                raise ValueError('side branch did not learn')
            if current['parameters'] != info['parameters'] or current['trainable_parameters'] != info['trainable_parameters']:
                raise ValueError('parameter budget changed')
            hashes[str(final.relative_to(root))] = file_sha(final)
            proofs.append({'seed': seed, 'kind': kind, 'paired_steps': steps, 'frozen_backbone_exact': True,
                           'parameters': current['parameters'], 'trainable_parameters': current['trainable_parameters']})
    proof = {'complete': True, 'seeds': seeds, 'steps': steps, 'proofs': proofs,
             'final_checkpoints_sha256': hashes, 'paired_teacher_case_query_video_sequences': True}
    save_json(output if output is not None else root / 'training-audit.json', proof)
    print(json.dumps(proof), flush=True)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--steps', type=int, required=True)
    a = p.parse_args()
    audit(a.root, [int(x) for x in a.seeds.split(',')], a.steps)
