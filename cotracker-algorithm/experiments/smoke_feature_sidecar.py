"""Load trained checkpoints with the actual inference loader and execute on GPU."""
import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker')]
from resources.model import setup_model
from cotracker.models.core.cotracker.feature_sidecar import KINDS


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise ValueError('GPU smoke requires working CUDA, not just nvidia-smi')
    torch.manual_seed(20261008)
    video = torch.rand(1, 23, 3, 64, 64, device='cuda') * 255
    queries = torch.tensor([[[0.,16.,16.], [5.,32.,32.], [10.,48.,48.]]], device='cuda')
    for kind in KINDS:
        model = setup_model(str(a.root / kind / 'cotracker_three_final.pth'), device='cuda')
        if model.feature_sidecar_kind != kind:
            raise ValueError('inference checkpoint kind differs')
        d = {}
        model.updateformer.feature_sidecar.diagnostics = d
        with torch.no_grad():
            out = model(video=video, queries=queries, iters=2)
        if any(not bool(torch.isfinite(x).all()) for x in out[:3]) or d.get('sidecar_calls') != 2:
            raise ValueError('invalid inference outputs / inactive branch')
        print('GPU INFERENCE SMOKE PASSED ' + kind + ' ' + str(d), flush=True)
        del model
        torch.cuda.empty_cache()
