"""Experimental Mamba-style replacement for UpdateFormer temporal attention."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MambaTimeBlock(nn.Module):
    """Bidirectional selective SSM with the same input/output contract as AttnBlock."""

    def __init__(self, hidden_size: int, state_dim: int = 8, expansion: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.inner = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, self.inner * 2)
        self.conv = nn.Conv1d(
            self.inner, self.inner, kernel_size=3, padding=2, groups=self.inner
        )
        self.parameter_proj = nn.Linear(self.inner, 1 + 2 * state_dim)
        self.dt_proj = nn.Linear(1, self.inner)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .view(1, -1)
            .repeat(self.inner, 1)
        )
        self.skip = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, hidden_size)
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        x, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)
        dt_raw, b_values, c_values = torch.split(
            self.parameter_proj(x), [1, self.state_dim, self.state_dim], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw) + 1e-4)
        a = -torch.exp(self.a_log).to(dtype=x.dtype)
        state = x.new_zeros((x.shape[0], self.inner, self.state_dim))
        output = []
        for index in range(length):
            dt_t = dt[:, index]
            state = (
                torch.exp(dt_t[..., None] * a[None]) * state
                + dt_t[..., None]
                * b_values[:, index, None, :]
                * x[:, index, :, None]
            )
            y = (state * c_values[:, index, None, :]).sum(dim=-1)
            output.append(y + self.skip * x[:, index])
        return self.out_proj(torch.stack(output, dim=1) * F.silu(gate))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        del mask
        forward = self._scan(x)
        backward = torch.flip(self._scan(torch.flip(x, dims=(1,))), dims=(1,))
        return x + self.output_scale * 0.5 * (forward + backward)


def replace_updateformer_time_attention(
    model: nn.Module,
    state_dim: int = 8,
    expansion: int = 1,
) -> list[nn.Parameter]:
    """Replace every temporal attention block and freeze all other parameters."""
    updateformer = model.updateformer
    count = len(updateformer.time_blocks)
    if count < 1:
        raise ValueError("UpdateFormer has no temporal blocks to replace")
    replacement = nn.ModuleList(
        [
            MambaTimeBlock(
                hidden_size=updateformer.hidden_size,
                state_dim=state_dim,
                expansion=expansion,
            )
            for _ in range(count)
        ]
    ).to(next(model.parameters()).device)
    updateformer.time_blocks = replacement
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in replacement.parameters():
        parameter.requires_grad = True
    return list(replacement.parameters())


class MambaTrajectoryRefiner(nn.Module):
    """Small bidirectional Mamba adapter that predicts bounded point residuals."""

    def __init__(
        self,
        model_dim: int = 24,
        state_dim: int = 8,
        blocks: int = 2,
        max_residual_pixels: float = 8.0,
    ) -> None:
        super().__init__()
        self.max_residual_pixels = max_residual_pixels
        self.input_proj = nn.Linear(6, model_dim)
        self.blocks = nn.ModuleList(
            [MambaTimeBlock(model_dim, state_dim=state_dim, expansion=2) for _ in range(blocks)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, 2)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        trajectories: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, points, coordinates = trajectories.shape
        if coordinates != 2:
            raise ValueError("trajectory coordinates must be two-dimensional")
        height, width = image_shape
        scale = trajectories.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        positions = trajectories / scale
        velocity = torch.zeros_like(positions)
        velocity[:, 1:] = positions[:, 1:] - positions[:, :-1]
        features = torch.cat(
            (positions, velocity, visibility[..., None], confidence[..., None]), dim=-1
        )
        hidden = self.input_proj(
            features.permute(0, 2, 1, 3).reshape(batch * points, frames, 6)
        )
        for block in self.blocks:
            hidden = block(hidden)
        residual = torch.tanh(self.output_proj(self.norm(hidden)))
        residual = residual.reshape(batch, points, frames, 2).permute(0, 2, 1, 3)
        residual = residual * self.max_residual_pixels
        refined = trajectories + residual
        refined_x = refined[..., 0].clamp(0, width - 1)
        refined_y = refined[..., 1].clamp(0, height - 1)
        refined = torch.stack((refined_x, refined_y), dim=-1)
        return refined, residual


def attach_trajectory_mamba_refiner(model: nn.Module) -> list[nn.Parameter]:
    """Attach a residual Mamba adapter and freeze the original CoTracker."""
    refiner = MambaTrajectoryRefiner().to(next(model.parameters()).device)
    model.trajectory_mamba_refiner = refiner
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in refiner.parameters():
        parameter.requires_grad = True
    return list(refiner.parameters())
