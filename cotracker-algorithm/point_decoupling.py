"""Fixed index selection and checked trajectory caches for point-count controls."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def subset_indices(parent, count, device=None):
    """Round evenly spaced indices, retaining the baseline's two endpoints.

    The baseline includes a duplicated closed-contour endpoint. Preserve that
    convention rather than quietly changing the polygon while testing density.
    """
    if not 3 <= count <= parent:
        raise ValueError('subset count must be within [3, parent]')
    i = torch.arange(count, dtype=torch.long, device=device)
    return (i * (parent - 1) + (count - 1) // 2) // (count - 1)


def array_sha(array):
    array = np.ascontiguousarray(array)
    header = json.dumps([str(array.dtype), list(array.shape)], separators=(',', ':'))
    return hashlib.sha256(header.encode() + array.tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_cache(path, *, profile, config, queries, parent_queries, trajectories,
               frames, target, model_shape, native_shape):
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in {
        'queries': queries, 'parent_queries': parent_queries,
        'trajectories': trajectories,
    }.items()}
    metadata = {
        'schema': 1, 'profile': profile, 'config': config,
        'model_shape': list(model_shape), 'native_shape': list(native_shape),
        'input_frames_sha256': array_sha(frames), 'input_target_sha256': array_sha(target),
        'array_hashes': {key: array_sha(value) for key, value in arrays.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.npz.tmp')
    with pending.open('wb') as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return metadata


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        arrays = {key: archive[key].copy() for key in ('queries', 'parent_queries', 'trajectories')}
    if metadata['schema'] != 1:
        raise ValueError('unsupported point cache schema')
    for key, array in arrays.items():
        if not np.isfinite(array).all() or array_sha(array) != metadata['array_hashes'][key]:
            raise ValueError('invalid point cache array: ' + key)
    queries, parent, trajectory = (arrays[key] for key in ('queries', 'parent_queries', 'trajectories'))
    if queries.ndim != 3 or queries.shape[0] != 1 or queries.shape[-1] != 3:
        raise ValueError('invalid query shape')
    if parent.ndim != 3 or parent.shape[0] != 1 or parent.shape[-1] != 3 or parent.shape[1] < queries.shape[1]:
        raise ValueError('invalid parent query shape')
    if trajectory.ndim != 4 or trajectory.shape[0] != 1 or trajectory.shape[1] < 1 or trajectory.shape[2:] != (queries.shape[1], 2):
        raise ValueError('invalid trajectory shape')
    config = metadata['config']
    if queries.shape[1] != config['border_points'] or parent.shape[1] != (config['query_parent_points'] or config['border_points']):
        raise ValueError('cache point counts differ from the executed config')
    if any(len(metadata[key]) != 2 or min(metadata[key]) <= 0 for key in ('model_shape', 'native_shape')):
        raise ValueError('invalid cached spatial shape')
    if not np.array_equal(trajectory[:, 0], queries[:, :, 1:]):
        raise ValueError('query-frame trajectory differs from query coordinates')
    return metadata, arrays
