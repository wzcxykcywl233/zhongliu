"""Explicit external-stage trainability and downstream handoff contract."""
import argparse
import os
from pathlib import Path
import sys

from acdc_data import save

PREFIX = 'updateformer.feature_sidecar.'
ROUTES = ('A1', 'B1', 'B2', 'B3')


def configure(model, route):
    from cotracker.models.core.cotracker.feature_sidecar import (
        attach_sidecar, configure_sidecar_runtime, set_sidecar_training_mode)
    if route not in ROUTES or hasattr(model.updateformer, 'feature_sidecar'):
        raise ValueError('Require plain pretrained backbone and known A/B route')
    model.frozen_pretrained = False
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(route == 'B3' or route == 'B1' and name.startswith('fnet.') or
                                 route == 'B2' and name.startswith('updateformer.'))
    if route == 'A1':
        attach_sidecar(model, 'mamba')
        configure_sidecar_runtime(model, backend='compact')
        set_sidecar_training_mode(model)
    elif route == 'B1':
        model.fnet.train()
    elif route == 'B2':
        model.updateformer.train()
    else:
        model.train()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('No trainable parameters')
    return {'route': route, 'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable_names': trainable, 'sidecar_attached': route == 'A1',
            'correlation_mlp_trainable': route == 'B3',
            'note': 'B2 includes coordinate/V/C heads within updateformer; corr_mlp stays frozen.'}


def check_routes(output):
    import torch
    from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
    torch.set_num_threads(1)
    reports = []
    for route in ROUTES:
        torch.manual_seed(20261023)
        model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        report = configure(model, route)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        trainable_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        video = torch.rand(1, 3, 3, 64, 64) * 255
        queries = torch.tensor([[[0., 20., 20.], [1., 40., 40.]]])
        p, v, c = model(video=video, queries=queries, iters=2, is_train=True)[:3]
        loss = p.square().mean() * .001 + v.square().mean() + c.square().mean()
        loss.backward()
        gradients = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        if not gradients or any(n in frozen for n in gradients):
            raise ValueError('Wrong/no gradient path: ' + route)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite gradient: ' + route)
        optimizer.step()
        state = dict(model.named_parameters())
        if any(not torch.equal(p, state[n]) for n, p in frozen.items()):
            raise ValueError('Frozen parameters changed: ' + route)
        if not any(not torch.equal(p, state[n]) for n, p in trainable_before.items()):
            raise ValueError('Optimizer did not update: ' + route)
        report.update(frozen_parameters_exact=True, gradient_path_verified=True,
                      updated_parameters_verified=True, gradient_names=gradients)
        reports.append(report)
        print('PARAMETER/GRADIENT CHECK PASSED ' + route, flush=True)
        del model, optimizer, frozen, trainable_before
    save(output / 'ab-parameter-check.json', {'complete': True, 'reports': reports,
         'scope': 'CPU random-weight code test; NOT medical pretraining or pretrained-weight GPU smoke'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path, default=Path(os.environ.get('COTRACKER_APP_DIR', '/checkout/cotracker-algorithm')))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.app / 'ext/co-tracker'))
    check_routes(args.output)
