"""Warm-start deeper CoTracker3 UpdateFormers without changing the base prefix."""
import copy
import json
from pathlib import Path

import torch

ARCHITECTURES = ('base', 'time6', 'space_time6')
PARAMETERS = {'base': 25385700, 'time6': 30704484, 'space_time6': 46665444}


def state_from_checkpoint(checkpoint):
    state = checkpoint
    if isinstance(state, dict):
        state = state.get('model_state_dict', state.get('model', state))
    if not isinstance(state, dict):
        raise ValueError('checkpoint must contain a model state dict')
    return {key.removeprefix('module.'): value for key, value in state.items()}


def identity_block(block):
    """Keep trained inner projections, zero the two residual output branches."""
    result = copy.deepcopy(block)
    attention = getattr(result, 'attn', getattr(result, 'cross_attn', None))
    with torch.no_grad():
        for layer in (attention.to_out, result.mlp.fc2):
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
    return result


def grow_backbone(model, architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError('unknown backbone architecture: ' + architecture)
    former = model.updateformer
    if len(former.time_blocks) != 3 or len(former.space_virtual_blocks) != 3:
        raise ValueError('growth requires the unchanged three-layer CoTracker3 base')
    if architecture != 'base':
        for _ in range(3):
            former.time_blocks.append(identity_block(former.time_blocks[2]))
        if architecture == 'space_time6':
            for name in ('space_virtual_blocks', 'space_virtual2point_blocks', 'space_point2virtual_blocks'):
                blocks = getattr(former, name)
                for _ in range(3):
                    blocks.append(identity_block(blocks[2]))
        # Original pretrained time/spatial stages remain at indices 0,1,2.
        # The old uniform-spacing rule would move them when time depth changes.
        former.space_layer_schedule = {i: i for i in range(len(former.space_virtual_blocks))}
    model.backbone_architecture = architecture
    return model


def _warm_start(model, checkpoint, architecture):
    model.load_state_dict(state_from_checkpoint(checkpoint), strict=True)
    mode = model.training
    model.eval()
    generator = torch.Generator().manual_seed(20261005)
    # CPU proof with real image features, correlations and two update iterations.
    # A private generator leaves the experiment's training RNG streams intact.
    video = torch.rand(1, 3, 3, 64, 64, generator=generator) * 255
    queries = torch.tensor([[[0., 16., 16.], [0., 32., 32.], [0., 48., 48.]]])
    with torch.no_grad():
        original = tuple(value.clone() for value in model(video=video, queries=queries, iters=2)[:3])
        grow_backbone(model, architecture)
        grown = model(video=video, queries=queries, iters=2)[:3]
    differences = [float((a - b).abs().max()) for a, b in zip(original, grown)]
    if any(value != 0. for value in differences):
        raise ValueError('growth initialization changed the base outputs: ' + str(differences))
    model.train(mode)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETERS[architecture]:
        raise ValueError('unexpected parameter count for ' + architecture)
    report = {'schema': 1, 'architecture': architecture, 'parameters': count,
              'added_parameters': count - PARAMETERS['base'],
              'initialization_max_abs_difference': differences,
              'initialization_outputs_exact': True,
              'time_layers': len(model.updateformer.time_blocks),
              'space_stages': len(model.updateformer.space_virtual_blocks),
              'space_schedule': {str(i): i for i in range(len(model.updateformer.space_virtual_blocks))},
              'initialization': 'pretrained prefix preserved; added attention and MLP output projections zeroed'}
    return report


def warm_start(model, checkpoint, architecture):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _warm_start(model, checkpoint, architecture)
    finally:
        torch.set_num_threads(threads)


def repair_audit_tail(path):
    """Preserve a partial last record before resuming an append-only audit."""
    import hashlib
    import os
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b'\n'):
        return
    boundary = data.rfind(b'\n') + 1
    tail = data[boundary:]
    recovery = path.with_name(path.name + '.tail-' + hashlib.sha256(tail).hexdigest()[:12] + '.txt')
    with recovery.open('wb') as stream:
        stream.write(tail)
        stream.flush()
        os.fsync(stream.fileno())
    with path.open('r+b') as stream:
        stream.truncate(boundary)
        stream.flush()
        os.fsync(stream.fileno())


def save_json(path, value):
    import os
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
