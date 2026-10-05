"""Check the reused environment and current training imports without a network."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ext' / 'co-tracker'))


def check(output):
    import torch
    import train_on_real_data  # Import all dependencies used by the actual trainer.
    from cotracker.models.core.cotracker.backbone_growth import ARCHITECTURES, PARAMETERS, save_json
    if not callable(train_on_real_data.fetch_optimizer):
        raise ValueError('current training entry point is unavailable')
    report = {
        'complete': True,
        'python': sys.version,
        'torch': str(torch.__version__),
        'torch_cuda': torch.version.cuda,
        'pytorch_lightning': importlib.metadata.version('pytorch-lightning'),
        'architectures': list(ARCHITECTURES),
        'parameters': PARAMETERS,
        'trainer': str(Path(train_on_real_data.__file__).resolve()),
        'note': 'CPU dependency check only; the separate GPU smoke stage verifies training.',
    }
    save_json(output, report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    check(parser.parse_args().output)
