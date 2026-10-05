"""Reconstruct a sparse polygon from the committed dense trajectory cache."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import POINT_DECOUPLING_EXPERIMENTS
from point_decoupling import array_sha, file_sha, load_cache, subset_indices


def reconstruct(source, frames, target):
    import resources
    metadata, arrays = load_cache(source)
    if metadata['profile'] != 'pd_dense' or metadata['config'] != asdict(POINT_DECOUPLING_EXPERIMENTS['pd_dense']):
        raise ValueError('replay requires a matching pd_dense cache')
    if array_sha(frames) != metadata['input_frames_sha256'] or array_sha(target) != metadata['input_target_sha256']:
        raise ValueError('replay cache does not match this input case')
    config = POINT_DECOUPLING_EXPERIMENTS['pd_dense_thin']
    if arrays['trajectories'].shape[2] != config.border_points:
        raise ValueError('unexpected dense tracking point count')
    indices = subset_indices(config.border_points, config.reconstruction_points).numpy()
    trajectories = torch.from_numpy(arrays['trajectories'][:, :, indices].copy())
    diagnostics = {
        'point_tracking_count': config.border_points,
        'point_reconstruction_count': config.reconstruction_points,
        'point_cache_replay': 1,
    }
    masks = resources.convert_point_trajectory_to_mask_sequence(
        trajectories, video_shape=tuple(metadata['model_shape']),
        morph_close_kernel=config.morph_close_kernel,
        keep_largest_component=config.keep_largest_component, diagnostics=diagnostics,
    )
    prediction = resources.reshape_video(masks, target_shape=tuple(metadata['native_shape']))
    prediction = prediction[0].numpy().transpose(2, 1, 0)
    if prediction.shape != frames.shape:
        raise ValueError('replayed mask has incorrect native dimensions')
    config_dict = asdict(config)
    config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    record = {
        'schema_version': 1, 'profile': 'pd_dense_thin', 'config': config_dict,
        'config_sha256': hashlib.sha256(config_json.encode()).hexdigest(),
        'mechanism': diagnostics,
        'point_decoupling': {
            'source_profile': 'pd_dense', 'source_cache_sha256': file_sha(source),
            'source_trajectory_sha256': metadata['array_hashes']['trajectories'],
            'selected_query_sha256': array_sha(arrays['queries'][:, indices]),
            'selected_trajectory_sha256': array_sha(trajectories.numpy()),
            'selected_indices': indices.tolist(),
        },
        'prediction': {
            'shape': list(prediction.shape), 'foreground_voxels': int(np.count_nonzero(prediction)),
            'array_sha256': hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),
        },
    }
    return prediction, record


def run():
    from inference import INPUT_PATH, OUTPUT_PATH, load_image_file_as_array, write_array_as_image_file
    if os.environ.get('COTRACKER_EXPERIMENT') != 'pd_dense_thin':
        raise ValueError('cache replay is reserved for pd_dense_thin')
    start = time.perf_counter()
    frames = load_image_file_as_array(location=INPUT_PATH / 'images/mri-linacs')
    target = load_image_file_as_array(location=INPUT_PATH / 'images/mri-linac-target')
    print(f'Runtime loading:   {time.perf_counter() - start:.5f} s', flush=True)
    start = time.perf_counter()
    prediction, record = reconstruct(Path('/point-source/point-cache.npz'), frames, target)
    print('TRACKRAD_DIAGNOSTICS_JSON=' + json.dumps(record, sort_keys=True, separators=(',', ':')), flush=True)
    print(f'Runtime algorithm: {time.perf_counter() - start:.5f} s', flush=True)
    start = time.perf_counter()
    write_array_as_image_file(location=OUTPUT_PATH / 'images/mri-linac-series-targets', array=prediction.astype(np.uint8))
    print(f'Runtime writing:   {time.perf_counter() - start:.5f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
