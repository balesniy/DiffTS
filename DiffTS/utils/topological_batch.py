from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TopologicalBatch:
    scan_points: list[torch.Tensor]
    scan_semantics: list[torch.Tensor]
    nodes_gt: torch.Tensor
    node_exists: torch.Tensor
    parent_ids: torch.Tensor
    highres_gt_nodes: list[torch.Tensor]
    highres_gt_edges: list[torch.Tensor]
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    filenames: list[str]
    normalization_factors: list[Any]


def binary_classes_to_semantics(classes: torch.Tensor) -> torch.Tensor:
    """Convert current binary scan labels into wood/leaf/unknown probabilities."""
    classes = classes.squeeze(-1) if classes.dim() > 1 and classes.shape[-1] == 1 else classes
    leaf = classes.float().clamp(0.0, 1.0)
    wood = 1.0 - leaf
    unknown = (1.0 - (wood + leaf).clamp(0.0, 1.0)).clamp_min(0.0)
    return torch.stack((wood, leaf, unknown), dim=-1)


def _pad_nodes(nodes: torch.Tensor, n_max: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_real = min(nodes.shape[0], n_max)
    out = nodes.new_zeros((n_max, 3))
    exists = torch.zeros((n_max,), dtype=torch.bool, device=nodes.device)
    out[:n_real] = nodes[:n_real]
    exists[:n_real] = True
    if n_real < n_max:
        out[n_real:] = torch.randn((n_max - n_real, 3), device=nodes.device, dtype=nodes.dtype)
    return out, exists, n_real


def _pad_parent_ids(parent_ids: torch.Tensor, n_max: int, n_real: int) -> torch.Tensor:
    out = torch.zeros((n_max,), dtype=torch.long, device=parent_ids.device)
    if n_real > 0:
        valid = parent_ids[:n_real].long().clamp(min=0, max=max(n_real - 1, 0))
        out[:n_real] = valid
    return out


def _bounds(points: torch.Tensor, nodes: torch.Tensor, padding: float) -> tuple[torch.Tensor, torch.Tensor]:
    all_points = torch.cat((points[:, :3], nodes[:, :3]), dim=0)
    bmin = all_points.amin(dim=0)
    bmax = all_points.amax(dim=0)
    extent = (bmax - bmin).clamp_min(1e-3)
    return bmin - padding * extent, bmax + padding * extent


def build_topological_batch(
    batch: dict,
    n_max: int,
    device: torch.device | str,
    bounds_padding: float = 0.05,
) -> TopologicalBatch:
    """Build the padded Topological-DDPM interface from the legacy collated batch."""
    scan_points: list[torch.Tensor] = []
    scan_semantics: list[torch.Tensor] = []
    highres_nodes: list[torch.Tensor] = []
    highres_edges: list[torch.Tensor] = []
    padded_nodes = []
    node_exists = []
    padded_parents = []
    bounds_min = []
    bounds_max = []

    for idx, nodes in enumerate(batch["pcd_nodes"]):
        nodes = nodes.to(device=device, dtype=torch.float32)
        points = batch["pcd_conditioning_pts"][idx].to(device=device, dtype=torch.float32)
        classes = batch["scan_point_classes"][idx].to(device=device)
        parents = batch["node_parent_ids"][idx].to(device=device)

        nodes_pad, exists, n_real = _pad_nodes(nodes, n_max)
        parents_pad = _pad_parent_ids(parents, n_max, n_real)
        bmin, bmax = _bounds(points, nodes[:n_real] if n_real > 0 else nodes_pad[:1], bounds_padding)

        scan_points.append(points[:, :3])
        scan_semantics.append(binary_classes_to_semantics(classes))
        highres_nodes.append(batch["full_pcd_nodes"][idx].to(device=device, dtype=torch.float32))
        highres_edges.append(batch["parent_ids"][idx].to(device=device, dtype=torch.long))
        padded_nodes.append(nodes_pad)
        node_exists.append(exists)
        padded_parents.append(parents_pad)
        bounds_min.append(bmin)
        bounds_max.append(bmax)

    return TopologicalBatch(
        scan_points=scan_points,
        scan_semantics=scan_semantics,
        nodes_gt=torch.stack(padded_nodes, dim=0),
        node_exists=torch.stack(node_exists, dim=0),
        parent_ids=torch.stack(padded_parents, dim=0),
        highres_gt_nodes=highres_nodes,
        highres_gt_edges=highres_edges,
        bounds_min=torch.stack(bounds_min, dim=0),
        bounds_max=torch.stack(bounds_max, dim=0),
        filenames=list(batch.get("filename", [""] * len(padded_nodes))),
        normalization_factors=list(batch.get("normalization_factors", [None] * len(padded_nodes))),
    )
