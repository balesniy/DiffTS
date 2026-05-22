from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class GuidanceConfig:
    active_fraction: float = 0.2
    final_decay_fraction: float = 0.05
    max_displacement: float = 0.02
    base_eta: float = 1.0
    repel_weight: float = 0.0
    repel_radius: float = 0.005
    repel_k: int = 8


def guidance_eta(step_index: int, total_steps: int, cfg: GuidanceConfig) -> float:
    if total_steps <= 1:
        return 0.0
    progress = step_index / float(total_steps - 1)
    start = max(0.0, 1.0 - cfg.active_fraction)
    if progress < start:
        return 0.0
    local = (progress - start) / max(cfg.active_fraction, 1e-6)
    if local > 1.0 - cfg.final_decay_fraction:
        tail = (1.0 - local) / max(cfg.final_decay_fraction, 1e-6)
        return cfg.base_eta * max(0.0, tail)
    return cfg.base_eta * math.sin(min(local, 1.0) * math.pi / 2.0)


def nodewise_clip(grad: torch.Tensor, max_displacement: float) -> torch.Tensor:
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    norm = grad.norm(dim=-1, keepdim=True)
    scale = torch.clamp(norm, max=max_displacement) / norm.clamp_min(1e-6)
    return grad * scale


def local_repulsion_loss(nodes: torch.Tensor, valid_mask: torch.Tensor, radius: float, k: int) -> torch.Tensor:
    batch, n_nodes, _ = nodes.shape
    if n_nodes <= 1:
        return nodes.new_zeros(())
    dist2 = torch.cdist(nodes, nodes).pow(2)
    eye = torch.eye(n_nodes, dtype=torch.bool, device=nodes.device).unsqueeze(0)
    valid = valid_mask[:, :, None] & valid_mask[:, None, :] & ~eye
    dist2 = dist2.masked_fill(~valid, torch.inf)
    knn = torch.topk(dist2, k=min(k, n_nodes - 1), dim=-1, largest=False).values
    penalty = F.relu((radius**2) / knn.clamp_min(1e-9) - 1.0)
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)
    return penalty.mean()


def guidance_displacement(
    x_t: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    step_index: int,
    total_steps: int,
    cfg: GuidanceConfig,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    eta = guidance_eta(step_index, total_steps, cfg)
    if eta <= 0.0:
        return torch.zeros_like(x_t)

    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        energy = energy_fn(x_in)
        if cfg.repel_weight > 0 and valid_mask is not None:
            energy = energy + cfg.repel_weight * local_repulsion_loss(
                x_in,
                valid_mask,
                radius=cfg.repel_radius,
                k=cfg.repel_k,
            )
        grad = torch.autograd.grad(energy, x_in, allow_unused=False)[0]

    return -eta * nodewise_clip(grad, cfg.max_displacement)
