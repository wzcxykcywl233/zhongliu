"""Same pretrained tensors, frozen computation flags, no optimizer/training."""
import argparse
from pathlib import Path
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ext' / 'co-tracker'), str(Path(__file__).parent)]
from audit_backbone_growth import file_sha
from cotracker.models.core.cotracker.backbone_growth import state_from_checkpoint
from cotracker.models.core.cotracker.feature_sidecar import tensor_hash


def freeze(source, output):
    source_sha = file_sha(source)
    state = state_from_checkpoint(torch.load(source, map_location='cpu', weights_only=True))
    if output.is_file():
        prior = torch.load(output, map_location='cpu', weights_only=True)
        if prior.get('source_checkpoint_sha256') != source_sha or not prior.get('freeze_base_for_inference') or tensor_hash(prior['model']) != tensor_hash(state):
            raise ValueError('frozen pretrained reference changed')
        return
    checkpoint = {'model':state, 'backbone_architecture':'base', 'freeze_base_for_inference':True,
                  'source_checkpoint_sha256':source_sha}
    temp = output.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temp, output)
    print('FROZEN PRETRAINED REFERENCE ' + str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    freeze(a.source, a.output)
