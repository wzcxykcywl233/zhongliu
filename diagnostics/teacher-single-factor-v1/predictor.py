"""Label-generator inference. No optimizer, student, future labels or Mamba."""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import numpy as np
import torch


def activate():
    source = Path(os.environ.get('COTRACKER_SOURCE_DIR', '/opt/app/ext/co-tracker')).resolve()
    if not (source / 'cotracker/__init__.py').is_file():
        raise ValueError('Copied CoTracker source missing')
    for name, module in tuple(sys.modules.items()):
        if name == 'cotracker' or name.startswith('cotracker.'):
            location = getattr(module, '__file__', None)
            if not location or source not in Path(location).resolve().parents:
                raise ValueError('Unverified CoTracker wheel already imported')
    sys.path[:0] = [str(source), str(source.parents[1])]
    importlib.invalidate_caches()
    return source


def tensor_sha(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def canonical(serialized):
    if serialized.ndim != 3 or not np.isfinite(serialized).all():
        raise ValueError('Expected finite serialized W,H,T input')
    return np.ascontiguousarray(serialized.transpose(2, 1, 0))


def preprocess(array, intensity, target_shape=(384, 512)):
    a = array.astype(np.float32, copy=True)
    if intensity == 'robust_1_99':
        low, high = np.percentile(a, (1, 99))
        if high <= low:
            raise ValueError('Non-identifiable intensity bounds')
        a = np.clip((a - low) / (high - low), 0, 1) * 255
    elif intensity != 'raw':
        raise ValueError('Unknown intensity factor')
    tensor = torch.from_numpy(a)[:, None]
    # Query XY is sampled in the model coordinate system. Resize video to the
    # SAME system; passing native-size MRI with 384x512 query XY is invalid.
    tensor = torch.nn.functional.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    return tensor[None]


def query_points(first, count):
    import resources
    # Float interpolation matches deployment; boolean interpolation shifts borders.
    mask = resources.reshape_video(torch.from_numpy(first.astype(np.float32))[None, None], (384, 512))
    queries = resources.convert_mask_to_points(mask, count).float()
    if queries.shape != (1, count, 3) or not torch.isfinite(queries).all():
        raise ValueError('Invalid fixed target queries')
    return queries


def uniform_support(video, seed):
    # Same fixed auxiliary support across teachers. They are not scored target IDs.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    h, w = video.shape[-2:]
    times = torch.randint(0, video.shape[1], (1, 100, 1), generator=generator).float()
    xy = torch.rand((1, 100, 2), generator=generator) * torch.tensor([w - 1, h - 1])
    return torch.cat((times, xy), -1)


def track(model, video, queries, config, seed, extra=None, device='cuda', resources_module=None):
    if resources_module is None:
        import resources as resources_module
    r = resources_module
    if queries.ndim != 3 or not torch.isfinite(queries).all():
        raise ValueError('Invalid target queries')
    # Existing hierarchical code resets query times to zero. Fail, never mislabel.
    if config['method'] == 'hierarchical' and bool((queries[..., 0] != 0).any()):
        raise ValueError('This teacher pilot only supports first-frame queries; arbitrary-time training needs a separate matched adapter')
    target_count = queries.shape[1]
    combined = queries.clone()
    if config['extra_border_points']:
        if extra is None or extra.shape != (1, config['extra_border_points'], 3):
            raise ValueError('Missing matched auxiliary border context')
        combined = torch.cat((combined, extra.clone()), 1)
    grid = {'none': 0, 'grid10': 10, 'uniform100': 0}[config['support']]
    if config['support'] == 'uniform100':
        if config['method'] != 'plain':
            raise ValueError('Uniform-time support is only specified for plain control')
        combined = torch.cat((combined, uniform_support(video, seed)), 1)
    diagnostics = {'target_points': target_count, 'extra_border_points': config['extra_border_points'],
                   'support_policy': config['support'], 'fixed_queries_sha256': tensor_sha(queries)}
    with torch.no_grad():
        if config['method'] == 'plain':
            result = r.forward_pass(model, video, combined, support_grid_size=grid,
                                    n_iterations=config['iterations'], device=device)
        elif config['method'] == 'hierarchical':
            result = r.hierarchical_forward_pass(
                model, video, combined, span=config['span'], support_grid_size=grid,
                n_iterations=config['iterations'], original_feature_weight=config['original_feature_weight'],
                dual_anchor_weight=config['dual_anchor_weight'], occlusion_merge=config['occlusion_merge'],
                occlusion_visibility_threshold=.5, occlusion_point_fraction=.5,
                query_memory_mode=config['memory_mode'],
                query_memory_slots=config['memory_slots'] if config['memory_mode'] != 'none' else 0,
                query_memory_min_reliability=.5, query_memory_min_similarity=.5,
                query_memory_original_floor=.3, query_memory_diversity_weight=.25,
                query_memory_refinement=config['refinement'], query_state_inheritance=config['state'],
                query_state_decay_tau=1., adaptive_anchor=config['adaptive_anchor'],
                anchor_relative_drop=.15, anchor_patience=2, anchor_max_age=30,
                diagnostics=diagnostics, device=device)
        else:
            raise ValueError('Unknown method')
    arrays = [result.trajectories[:, :, :target_count].detach().cpu(),
              result.visibility[:, :, :target_count].detach().cpu(),
              result.confidence[:, :, :target_count].detach().cpu()]
    if arrays[0].shape != (1, video.shape[1], target_count, 2) or any(
            x.shape != (1, video.shape[1], target_count) for x in arrays[1:]):
        raise ValueError('Teacher output shape mismatch')
    if not all(torch.isfinite(x).all() for x in arrays):
        raise ValueError('Nonfinite teacher output')
    if any(bool(((x < 0) | (x > 1)).any()) for x in arrays[1:]):
        raise ValueError('Teacher V/C outside [0,1]')
    return arrays, diagnostics


def label_diagnostics(arrays, height=384, width=512):
    p, v, c = [x.numpy() for x in arrays]
    p, v, c = p[:, 1:], v[:, 1:], c[:, 1:]
    out = (p[..., 0] < 0) | (p[..., 0] >= width) | (p[..., 1] < 0) | (p[..., 1] >= height)
    acceleration = np.linalg.norm(np.diff(p, n=2, axis=1), axis=-1)
    return {'visible_gt09_fraction': float((v > .9).mean()), 'visibility_mean': float(v.mean()),
            'confidence_mean': float(c.mean()), 'out_of_bounds_fraction': float(out.mean()),
            'acceleration_mean_pixels': float(acceleration.mean()) if acceleration.size else None,
            'note': 'V/C and smoothness are not calibrated accuracy or point correspondence ground truth.'}


def render(arrays, native_shape, first):
    import resources
    mask = resources.convert_point_trajectory_to_mask_sequence(arrays[0], video_shape=(384, 512))
    mask = resources.reshape_video(mask, native_shape)[0].numpy().astype(bool)
    mask[0] = first
    return mask


def load_teacher(path, device='cuda'):
    activate()
    from cotracker.models.build_cotracker import build_cotracker
    # Same offline architecture/window for both old and scaled weights, strict load.
    model = build_cotracker(str(path), offline=True, window_len=60)
    model.requires_grad_(False).eval().to(device)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Teacher must remain frozen/eval')
    return model
