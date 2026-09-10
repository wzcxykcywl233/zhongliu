"""Full-sequence learned fusion between hierarchical and global tracking."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cotracker.models.core.cotracker.mamba_time import MambaTimeBlock


LONG_FUSION_FEATURE_DIM = 17


def _motion_norm(trajectories: torch.Tensor, diagonal: float) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = torch.zeros_like(trajectories)
    velocity[:, 1:] = trajectories[:, 1:] - trajectories[:, :-1]
    acceleration = torch.zeros_like(velocity)
    acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
    velocity_norm = torch.linalg.vector_norm(velocity, dim=-1).mean(dim=-1) / diagonal
    acceleration_norm = (
        torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1) / diagonal
    )
    return velocity_norm, acceleration_norm


def build_long_fusion_features(
    hierarchical,
    global_result,
    hierarchical_similarity: torch.Tensor,
    global_similarity: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Build one compact state token per frame over the complete video."""
    height, width = image_shape
    diagonal = float((height * height + width * width) ** 0.5)
    disagreement = torch.linalg.vector_norm(
        hierarchical.trajectories - global_result.trajectories, dim=-1
    )
    hierarchical_velocity, hierarchical_acceleration = _motion_norm(
        hierarchical.trajectories, diagonal
    )
    global_velocity, global_acceleration = _motion_norm(
        global_result.trajectories, diagonal
    )
    hierarchical_centroid = hierarchical.trajectories.mean(dim=2)
    global_centroid = global_result.trajectories.mean(dim=2)
    hierarchical_shift = torch.linalg.vector_norm(
        hierarchical_centroid - hierarchical_centroid[:, :1], dim=-1
    ) / diagonal
    global_shift = torch.linalg.vector_norm(
        global_centroid - global_centroid[:, :1], dim=-1
    ) / diagonal
    frames = hierarchical.trajectories.shape[1]
    time = torch.linspace(
        0.0,
        1.0,
        frames,
        device=hierarchical.trajectories.device,
        dtype=hierarchical.trajectories.dtype,
    ).view(1, frames).expand(hierarchical.trajectories.shape[0], -1)
    features = torch.stack(
        (
            time,
            disagreement.mean(dim=-1) / diagonal,
            disagreement.max(dim=-1).values / diagonal,
            hierarchical.visibility.mean(dim=-1),
            global_result.visibility.mean(dim=-1),
            hierarchical.confidence.mean(dim=-1),
            global_result.confidence.mean(dim=-1),
            hierarchical_similarity.mean(dim=-1),
            global_similarity.mean(dim=-1),
            hierarchical_velocity,
            global_velocity,
            hierarchical_acceleration,
            global_acceleration,
            hierarchical_shift,
            global_shift,
            (hierarchical.visibility < 0.5).float().mean(dim=-1),
            (global_result.visibility < 0.5).float().mean(dim=-1),
        ),
        dim=-1,
    )
    if features.shape[-1] != LONG_FUSION_FEATURE_DIM:
        raise RuntimeError("unexpected long-fusion feature dimension")
    return features


class FrameMLPFusionGate(nn.Module):
    """Per-frame control with no temporal communication."""

    def __init__(self, input_dim: int = LONG_FUSION_FEATURE_DIM, model_dim: int = 48):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class MambaLongFusionGate(nn.Module):
    """Bidirectional Mamba-style gate over every frame in a complete case."""

    def __init__(
        self,
        input_dim: int = LONG_FUSION_FEATURE_DIM,
        model_dim: int = 48,
        state_dim: int = 8,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                MambaTimeBlock(model_dim, state_dim=state_dim, expansion=2)
                for _ in range(blocks)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden)).squeeze(-1)


def make_long_fusion_gate(kind: str) -> nn.Module:
    if kind == "mlp":
        return FrameMLPFusionGate()
    if kind == "mamba":
        return MambaLongFusionGate()
    raise ValueError(f"unsupported long fusion gate: {kind}")


def load_long_fusion_gate(
    path: str | Path,
    expected_kind: str,
    device: str | torch.device,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    kind = checkpoint.get("kind")
    if kind != expected_kind:
        raise ValueError(f"gate checkpoint is {kind}, expected {expected_kind}")
    model = make_long_fusion_gate(kind)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    mean = checkpoint["feature_mean"].to(device)
    std = checkpoint["feature_std"].to(device).clamp_min(1e-6)
    return model, mean, std


def apply_long_fusion_gate(
    hierarchical,
    global_result,
    logits: torch.Tensor,
):
    """Reliability-aware fusion using a learned per-frame global weight."""
    global_prior = torch.sigmoid(logits).unsqueeze(-1)
    local_score = hierarchical.visibility * hierarchical.confidence
    global_score = global_result.visibility * global_result.confidence
    local_weight = (1.0 - global_prior) * local_score
    global_weight = global_prior * global_score
    denominator = local_weight + global_weight
    fallback = (
        (1.0 - global_prior[..., None]) * hierarchical.trajectories
        + global_prior[..., None] * global_result.trajectories
    )
    trajectories = torch.where(
        denominator[..., None] > 1e-6,
        (
            local_weight[..., None] * hierarchical.trajectories
            + global_weight[..., None] * global_result.trajectories
        )
        / denominator.clamp_min(1e-6)[..., None],
        fallback,
    )
    visibility = (
        (1.0 - global_prior) * hierarchical.visibility
        + global_prior * global_result.visibility
    )
    confidence = (
        (1.0 - global_prior) * hierarchical.confidence
        + global_prior * global_result.confidence
    )
    trajectories[:, 0] = hierarchical.trajectories[:, 0]
    result_type = type(hierarchical)
    return result_type(trajectories, visibility, confidence), global_prior.squeeze(-1)
