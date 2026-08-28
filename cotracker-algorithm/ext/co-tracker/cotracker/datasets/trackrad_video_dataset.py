"""TrackRAD MHA videos as unlabeled clips for pseudo-label fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cotracker.datasets.utils import CoTrackerData


class TrackRADVideoDataset(torch.utils.data.Dataset):
    """Load TrackRAD cases while deliberately ignoring target masks.

    CoTracker's real-video training creates pseudo labels from teacher models,
    therefore only ``images/*_frames.mha`` is consumed.  The MHA time axis is
    the first SimpleITK image dimension and consequently the last NumPy axis.
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: tuple[int, int] | list[int] = (384, 512),
        seq_len: int = 10,
        traj_per_sample: int = 768,
        random_frame_rate: bool = False,
        limit_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"TrackRAD dataset directory not found: {self.root}")
        self.files = sorted(self.root.glob("*/images/*_frames.mha"))
        if limit_samples is not None and limit_samples > 0:
            self.files = self.files[:limit_samples]
        if not self.files:
            raise FileNotFoundError(
                f"No TrackRAD frame files matching */images/*_frames.mha in {self.root}"
            )
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.crop_size = tuple(crop_size)
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.random_frame_rate = random_frame_rate

    @staticmethod
    def _read_mha(path: Path) -> torch.Tensor:
        import SimpleITK as sitk

        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D TrackRAD video, got {array.shape}: {path}")
        # SimpleITK reverses the MetaImage dimensions. TrackRAD stores DimSize
        # as T,H,W, therefore GetArrayFromImage returns W,H,T.
        video = torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0)))
        lower = torch.quantile(video, 0.01)
        upper = torch.quantile(video, 0.99)
        video = ((video - lower) / (upper - lower).clamp_min(1e-6)).clamp(0, 1)
        return video.mul(255.0)

    def _sample_clip(self, video: torch.Tensor) -> torch.Tensor:
        while video.shape[0] < self.seq_len:
            video = torch.cat([video, video.flip(0)], dim=0)
        max_rate = max(1, video.shape[0] // self.seq_len)
        rate = 1
        if self.random_frame_rate and max_rate > 1:
            rate = int(torch.randint(1, min(4, max_rate) + 1, ()).item())
        required = self.seq_len * rate
        if video.shape[0] > required:
            start = int(torch.randint(0, video.shape[0] - required + 1, ()).item())
        else:
            start = 0
        return video[start : start + required : rate][: self.seq_len]

    def __getitem__(self, index: int):
        path = self.files[index]
        video = self._sample_clip(self._read_mha(path))
        video = F.interpolate(
            video[:, None],
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        ).expand(-1, 3, -1, -1).contiguous()
        frames = video.shape[0]
        sample = CoTrackerData(
            video=video,
            trajectory=torch.ones(frames, self.traj_per_sample, 2),
            visibility=torch.ones(frames, self.traj_per_sample),
            valid=torch.ones(frames, self.traj_per_sample),
            seq_name=path.parent.parent.name,
        )
        return sample, True

    def __len__(self) -> int:
        return len(self.files)
