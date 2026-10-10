"""Check the REAL model entry import path before running validation cases."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Do not add ext/co-tracker here: model.py itself must select it, like inference.py.
sys.path.insert(0, str(ROOT))
import model
import torch
from cotracker.models.core.cotracker import cotracker3_offline, feature_sidecar


def preflight(training_root):
    source = (ROOT / 'ext' / 'co-tracker').resolve()
    for module in (cotracker3_offline, feature_sidecar):
        if source not in Path(module.__file__).resolve().parents:
            raise ValueError('inference imports stale installed code')
        print('INFERENCE SOURCE ' + str(module.__file__), flush=True)
    if not torch.cuda.is_available():
        raise ValueError('GPU inference preflight requires CUDA')
    torch.manual_seed(20261008)
    video = torch.rand(1,23,3,64,64,device='cuda') * 255
    queries = torch.tensor([[[0.,16.,16.],[5.,32.,32.],[10.,48.,48.]]],device='cuda')
    for kind in feature_sidecar.KINDS:
        tracker = model.resources.setup_model(str(training_root / 'seed_0' / kind / 'cotracker_three_final.pth'), device='cuda')
        diagnostics = {}
        tracker.updateformer.feature_sidecar.diagnostics = diagnostics
        with torch.no_grad():
            result = tracker(video=video, queries=queries, iters=2)
        if tracker.feature_sidecar_kind != kind or any(not bool(torch.isfinite(x).all()) for x in result[:3]):
            raise ValueError('invalid inference checkpoint/results: ' + kind)
        if diagnostics.get('sidecar_calls') != 2 or not diagnostics.get('sidecar_residual_rms_max', 0):
            raise ValueError('inactive learned sidecar: ' + kind)
        print('REAL ENTRY GPU PREFLIGHT PASSED ' + kind, flush=True)
        del tracker
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, required=True)
    preflight(p.parse_args().train)
