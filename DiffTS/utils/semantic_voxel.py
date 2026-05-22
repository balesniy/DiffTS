from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F


@dataclass
class SemanticVoxelGrid:
    grid: torch.Tensor
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor

    @classmethod
    def from_points(
        cls,
        scan_points: list[torch.Tensor],
        scan_semantics: list[torch.Tensor],
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        grid_size: Sequence[int] = (128, 128, 256),
        blur_sigma: float = 1.0,
    ) -> "SemanticVoxelGrid":
        grids = []
        for pts, sem, bmin, bmax in zip(scan_points, scan_semantics, bounds_min, bounds_max):
            grids.append(_rasterize_one(pts, sem, bmin, bmax, grid_size))
        grid = torch.stack(grids, dim=0)
        if blur_sigma > 0:
            grid = gaussian_blur_3d(grid, blur_sigma)
        return cls(grid=grid, bounds_min=bounds_min, bounds_max=bounds_max)

    def normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        bmin = self.bounds_min
        bmax = self.bounds_max
        while bmin.dim() < coords.dim():
            bmin = bmin.unsqueeze(1)
            bmax = bmax.unsqueeze(1)
        scale = (bmax - bmin).clamp_min(1e-6)
        return 2.0 * (coords - bmin) / scale - 1.0

    def sample(self, coords: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample semantic channels at [B, ..., 3] coordinates."""
        batch = coords.shape[0]
        original_shape = coords.shape[1:-1]
        flat_coords = coords.reshape(batch, -1, 3)
        grid = self.normalize_coords(flat_coords).view(batch, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.grid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).squeeze(-1).transpose(1, 2)
        return sampled.reshape(batch, *original_shape, self.grid.shape[1])


def _rasterize_one(
    points: torch.Tensor,
    semantics: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    grid_size: Sequence[int],
) -> torch.Tensor:
    channels = semantics.shape[-1]
    depth, height, width = [int(v) for v in grid_size]
    grid = points.new_zeros((channels, depth, height, width))
    counts = points.new_zeros((1, depth, height, width))

    coords = (points - bounds_min) / (bounds_max - bounds_min).clamp_min(1e-6)
    coords = coords.clamp(0.0, 0.999999)
    ix = (coords[:, 0] * width).long()
    iy = (coords[:, 1] * height).long()
    iz = (coords[:, 2] * depth).long()
    linear = iz * (height * width) + iy * width + ix

    flat_grid = grid.view(channels, -1)
    flat_counts = counts.view(1, -1)
    flat_grid.scatter_add_(1, linear.unsqueeze(0).expand(channels, -1), semantics.transpose(0, 1))
    flat_counts.scatter_add_(1, linear.unsqueeze(0), torch.ones_like(linear, dtype=points.dtype).unsqueeze(0))
    grid = flat_grid / flat_counts.clamp_min(1.0)
    return grid.view(channels, depth, height, width)


def gaussian_blur_3d(grid: torch.Tensor, sigma: float, kernel_size: int | None = None) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    radius = kernel_size // 2
    axis = torch.arange(-radius, radius + 1, device=grid.device, dtype=grid.dtype)
    kernel_1d = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = grid.shape[1]
    out = grid
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = kernel_size
        weight = kernel_1d.view(*shape).expand(channels, 1, -1 if dim == 0 else 1, -1 if dim == 1 else 1, -1 if dim == 2 else 1)
        out = F.conv3d(out, weight, padding=[radius if i == dim else 0 for i in range(3)], groups=channels)
    return out


def softmin(values: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    weights = torch.softmax(-values / max(temperature, 1e-6), dim=dim)
    return (weights * values).sum(dim=dim)


def corridor_support(
    voxel_grid: SemanticVoxelGrid,
    nodes: torch.Tensor,
    edge_index: torch.Tensor,
    num_samples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return wood/leaf/unknown support for directed edges [B,E,2] as parent->child."""
    batch, edges, _ = edge_index.shape
    batch_idx = torch.arange(batch, device=nodes.device).view(batch, 1).expand(batch, edges)
    parent = nodes[batch_idx, edge_index[..., 0]]
    child = nodes[batch_idx, edge_index[..., 1]]
    alphas = torch.linspace(0.0, 1.0, num_samples, device=nodes.device, dtype=nodes.dtype).view(1, 1, -1, 1)
    samples = parent.unsqueeze(2) + alphas * (child - parent).unsqueeze(2)
    profile = voxel_grid.sample(samples)
    wood = profile[..., 0]
    leaf = profile[..., 1] if profile.shape[-1] > 1 else torch.zeros_like(wood)
    unknown = profile[..., 2] if profile.shape[-1] > 2 else torch.zeros_like(wood)
    c_ij = 0.7 * wood.mean(dim=-1) + 0.3 * softmin(wood, dim=-1)
    return c_ij, leaf.mean(dim=-1), unknown.mean(dim=-1)
