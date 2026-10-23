"""Real native-time 2D cine clips, patient-balanced and restart deterministic."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from acdc_data import SCHEMA, digest, read


def sample_cine(array, seed, frames=16, resize=(384, 512)):
    if array.ndim != 4 or array.shape[0] < frames or not np.isfinite(array).all():
        raise ValueError('Expected finite T,Z,Y,X with enough native frames')
    generator = np.random.default_rng(seed)
    z = int(generator.integers(array.shape[1]))
    start = int(generator.integers(array.shape[0] - frames + 1))
    plane = array[:, z].astype(np.float32)
    low, high = (float(x) for x in np.percentile(plane, [1, 99]))
    if high <= low:
        low, high = float(plane.min()), float(plane.max())
    if high <= low:
        raise ValueError('Constant slice; inspect patient rather than silently skipping')
    selected = plane[start:start+frames].copy()
    selected -= np.float32(low)
    selected /= np.float32(high - low)
    np.clip(selected, 0, 1, out=selected)
    selected *= np.float32(255)
    video = F.interpolate(torch.from_numpy(selected)[:, None], size=resize,
                          mode='bilinear', align_corners=False)
    # Bilinear interpolation may overshoot by a few float32 ULPs only.
    if not torch.isfinite(video).all() or video.min() < -1e-4 or video.max() > 255.0001:
        raise ValueError('Unexpected resize range')
    video = video.clamp(0, 255).expand(-1, 3, -1, -1).contiguous()
    return video, {'slice_z': z, 'frames': list(range(start, start+frames)),
                   'intensity_low': low, 'intensity_high': high, 'source_shape_tzyx': list(array.shape)}


class ACDCCineDataset(torch.utils.data.Dataset):
    def __init__(self, root, manifest, split='external-train-80', seed=20261023):
        self.root, self.manifest_path = Path(root), Path(manifest)
        self.manifest = read(manifest)
        if self.manifest['schema'] != SCHEMA or not self.manifest['training_ready']:
            raise ValueError('Only the complete 100-patient audited manifest is trainable')
        if split not in ('external-train-80', 'external-dev-20'):
            raise ValueError('No external test access in this loader')
        self.rows = sorted((r for r in self.manifest['patients'] if r['split'] == split), key=lambda r: r['patient'])
        if len(self.rows) != (80 if split == 'external-train-80' else 20):
            raise ValueError('Invalid split size')
        self.seed, self.epoch = seed, 0
        # Once per dataset construction, before yielding any sample.
        for row in self.rows:
            path = self.root / row['cine']
            if digest(path) != row['files_sha256'][path.name]:
                raise ValueError('Cine changed after audit: ' + row['patient'])

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import SimpleITK as sitk
        row = self.rows[index]
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(self.root / row['cine'])))
        if list(array.shape) != row['shape_tzyx']:
            raise ValueError('Changed shape')
        video, proof = sample_cine(array, self.seed + self.epoch * 1000003 + index)
        return {'video': video, 'patient': row['patient'], 'group': row['group'], 'sampling': proof}
