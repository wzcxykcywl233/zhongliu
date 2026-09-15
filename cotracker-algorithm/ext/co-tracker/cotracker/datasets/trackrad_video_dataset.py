"""TrackRAD MHA clips for pseudo-label and optional mask-assisted training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cotracker.datasets.utils import CoTrackerData


class TrackRADVideoDataset(torch.utils.data.Dataset):
    """Load TrackRAD cases with optional paired target-mask sequences.

    The default remains teacher-only pseudo-label training.  Ground-truth masks
    are loaded only for explicitly configured mask-assisted experiments.  The
    MHA time axis is the first SimpleITK image dimension and consequently the
    last NumPy axis.
    """

    def __init__(
        self,
        root: str | Path,
        crop_size: tuple[int, int] | list[int] = (384, 512),
        seq_len: int = 10,
        traj_per_sample: int = 768,
        random_frame_rate: bool = False,
        limit_samples: int | None = None,
        load_target_masks: bool = False,
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
        self.load_target_masks = load_target_masks

    @staticmethod
    def _robust_intensity_bounds(
        array: np.ndarray,
        max_samples: int = 1_000_000,
    ) -> tuple[float, float]:
        """Estimate deterministic percentiles without sorting a huge tensor."""

        flat = array.reshape(-1)
        if flat.size > max_samples:
            stride = int(np.ceil(flat.size / max_samples))
            flat = flat[::stride]
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return 0.0, 1.0
        lower, upper = np.percentile(finite, (1.0, 99.0))
        if upper <= lower:
            lower = float(finite.min())
            upper = float(finite.max())
        if upper <= lower:
            upper = lower + 1.0
        return float(lower), float(upper)

    @staticmethod
    def _read_mha(path: Path) -> torch.Tensor:
        import SimpleITK as sitk

        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D TrackRAD video, got {array.shape}: {path}")
        lower, upper = TrackRADVideoDataset._robust_intensity_bounds(array)
        # SimpleITK reverses the MetaImage dimensions. TrackRAD stores DimSize
        # as T,H,W, therefore GetArrayFromImage returns W,H,T.
        video = torch.from_numpy(np.ascontiguousarray(np.moveaxis(array, -1, 0)))
        video = ((video - lower) / max(upper - lower, 1e-6)).clamp(0, 1)
        return video.mul(255.0)

    def _sample_indices(self, frame_count: int) -> torch.Tensor:
        indices = torch.arange(frame_count)
        while indices.shape[0] < self.seq_len:
            indices = torch.cat([indices, indices.flip(0)], dim=0)
        max_rate = max(1, indices.shape[0] // self.seq_len)
        rate = 1
        if self.random_frame_rate and max_rate > 1:
            rate = int(torch.randint(1, min(4, max_rate) + 1, ()).item())
        required = self.seq_len * rate
        if indices.shape[0] > required:
            start = int(torch.randint(0, indices.shape[0] - required + 1, ()).item())
        else:
            start = 0
        return indices[start : start + required : rate][: self.seq_len]

    def _sample_clip(self, video: torch.Tensor) -> torch.Tensor:
        return video[self._sample_indices(video.shape[0])]

    @staticmethod
    def _read_mask_mha(path: Path) -> torch.Tensor:
        import SimpleITK as sitk

        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D TrackRAD mask, got {array.shape}: {path}")
        return torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(array, -1, 0) > 0)
        )

    def __getitem__(self, index: int):
        path = self.files[index]
        full_video = self._read_mha(path)
        clip_indices = self._sample_indices(full_video.shape[0])
        video = full_video[clip_indices]
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
        if self.load_target_masks:
            case_id = path.parent.parent.name
            mask_path = path.parent.parent / "targets" / f"{case_id}_labels.mha"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask-assisted training requires full labels: {mask_path}"
                )
            full_masks = self._read_mask_mha(mask_path)
            if full_masks.shape != full_video.shape:
                raise ValueError(
                    "TrackRAD frame/mask shape mismatch: "
                    f"{full_video.shape} vs {full_masks.shape} in {case_id}"
                )
            sample.segmentation = F.interpolate(
                full_masks[clip_indices, None].float(),
                size=self.crop_size,
                mode="nearest",
            )
        return sample, True

    def __len__(self) -> int:
        return len(self.files)
