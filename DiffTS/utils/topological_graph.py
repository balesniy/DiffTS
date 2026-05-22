from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from DiffTS.utils.semantic_voxel import SemanticVoxelGrid, corridor_support
from DiffTS.utils.utils import normalize_vecs


@dataclass
class SparseGraphOutput:
    edge_index: torch.Tensor
    edge_logits: torch.Tensor
    edge_weights: torch.Tensor
    edge_features: torch.Tensor
    edge_valid: torch.Tensor
    corridor_wood: torch.Tensor
    corridor_leaf: torch.Tensor
    corridor_unknown: torch.Tensor


def logit_safe(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    values = values.clamp(eps, 1.0 - eps)
    return torch.log(values / (1.0 - values))


def topk_candidate_edges(
    nodes: torch.Tensor,
    flow: torch.Tensor,
    node_support: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int = 12,
    sigma: float = 0.1,
    beta_flow: float = 1.0,
    beta_node: float = 0.5,
    child_chunk_size: int = 512,
) -> torch.Tensor:
    """Create sparse parent->child candidate edges without materializing downstream dense adjacency."""
    batch, n_nodes, _ = nodes.shape
    k = min(k, max(n_nodes - 1, 1))
    flow_n = normalize_vecs(flow.reshape(-1, 3)).view_as(flow)

    wood = node_support[..., 0].clamp(0.0, 1.0)
    unknown = node_support[..., 2].clamp(0.0, 1.0) if node_support.shape[-1] > 2 else torch.zeros_like(wood)
    wood_safe = (wood + unknown).clamp(1e-4, 1.0 - 1e-4)

    chunks = []
    parent_ids = torch.arange(n_nodes, device=nodes.device).view(1, 1, n_nodes)
    for start in range(0, n_nodes, child_chunk_size):
        end = min(start + child_chunk_size, n_nodes)
        child_ids = torch.arange(start, end, device=nodes.device).view(1, end - start, 1)
        child_to_parent = nodes[:, start:end, None, :] - nodes[:, None, :, :]
        dist2 = (child_to_parent**2).sum(dim=-1)
        dir_n = normalize_vecs(child_to_parent.reshape(-1, 3)).view_as(child_to_parent)
        flow_score = (flow_n[:, start:end, None, :] * dir_n).sum(dim=-1)
        node_score = logit_safe(torch.sqrt(wood_safe[:, start:end, None] * wood_safe[:, None, :]))
        pre_score = -dist2 / (2.0 * max(sigma, 1e-6) ** 2) + beta_flow * flow_score + beta_node * node_score
        valid_pair = valid_mask[:, start:end, None] & valid_mask[:, None, :] & (parent_ids != child_ids)
        pre_score = pre_score.masked_fill(~valid_pair, -torch.inf)
        parent_idx = torch.topk(pre_score, k=k, dim=-1).indices.detach()
        child_idx = child_ids.expand(batch, end - start, k)
        chunks.append(torch.stack((parent_idx, child_idx), dim=-1).reshape(batch, (end - start) * k, 2))
    return torch.cat(chunks, dim=1)


def build_edge_features(
    nodes: torch.Tensor,
    flow: torch.Tensor,
    depth: torch.Tensor,
    node_latents: torch.Tensor,
    edge_index: torch.Tensor,
    c_ij: torch.Tensor,
    l_ij: torch.Tensor,
    u_ij: torch.Tensor,
) -> torch.Tensor:
    batch, edges, _ = edge_index.shape
    batch_idx = torch.arange(batch, device=nodes.device).view(batch, 1).expand(batch, edges)
    parent_id = edge_index[..., 0]
    child_id = edge_index[..., 1]
    parent = nodes[batch_idx, parent_id]
    child = nodes[batch_idx, child_id]
    parent_h = node_latents[batch_idx, parent_id]
    child_h = node_latents[batch_idx, child_id]
    edge_vec = child - parent
    length = edge_vec.norm(dim=-1, keepdim=True)
    flow_align = (normalize_vecs(flow[batch_idx, child_id]) * normalize_vecs(parent - child)).sum(dim=-1, keepdim=True)
    delta_z = (depth[batch_idx, child_id] - depth[batch_idx, parent_id]).unsqueeze(-1)
    scalar_features = torch.cat(
        (
            length,
            flow_align,
            c_ij.unsqueeze(-1),
            l_ij.unsqueeze(-1),
            u_ij.unsqueeze(-1),
            delta_z,
        ),
        dim=-1,
    )
    return torch.cat((parent_h, child_h, scalar_features), dim=-1)


def soft_adjacency_from_logits(
    edge_logits: torch.Tensor,
    nodes: torch.Tensor,
    edge_index: torch.Tensor,
    c_ij: torch.Tensor,
    l_ij: torch.Tensor,
    tau: float = 0.25,
    beta_corr: float = 1.0,
    beta_leaf: float = 1.0,
    sigma: float = 0.1,
) -> torch.Tensor:
    batch, edges, _ = edge_index.shape
    batch_idx = torch.arange(batch, device=nodes.device).view(batch, 1).expand(batch, edges)
    parent = nodes[batch_idx, edge_index[..., 0]]
    child = nodes[batch_idx, edge_index[..., 1]]
    dist_penalty = ((child - parent) ** 2).sum(dim=-1) / (2.0 * max(sigma, 1e-6) ** 2)
    logits = edge_logits - dist_penalty + beta_corr * logit_safe(c_ij) - beta_leaf * l_ij
    return torch.sigmoid(logits / tau)


def depth_topological_loss(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    depth: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    batch, edges, _ = edge_index.shape
    batch_idx = torch.arange(batch, device=depth.device).view(batch, 1).expand(batch, edges)
    parent_depth = depth[batch_idx, edge_index[..., 0]]
    child_depth = depth[batch_idx, edge_index[..., 1]]
    penalty = F.relu(parent_depth - child_depth + margin).pow(2)
    return (edge_weights * penalty).mean()


def two_cycle_loss(edge_index: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
    losses = []
    for edges_b, weights_b in zip(edge_index, edge_weights):
        lookup = {(int(p), int(c)): idx for idx, (p, c) in enumerate(edges_b.tolist())}
        terms = []
        for idx, (p, c) in enumerate(edges_b.tolist()):
            rev = lookup.get((c, p))
            if rev is not None:
                terms.append(weights_b[idx] * weights_b[rev])
        if terms:
            losses.append(torch.stack(terms).mean())
        else:
            losses.append(weights_b.new_zeros(()))
    return torch.stack(losses).mean()


def build_sparse_graph(
    nodes: torch.Tensor,
    flow: torch.Tensor,
    depth: torch.Tensor,
    node_latents: torch.Tensor,
    valid_mask: torch.Tensor,
    voxel_grid: SemanticVoxelGrid,
    edge_mlp: torch.nn.Module,
    k: int = 12,
    corridor_samples: int = 8,
    sigma: float = 0.1,
    tau: float = 0.25,
) -> SparseGraphOutput:
    node_support = voxel_grid.sample(nodes)
    edge_index = topk_candidate_edges(nodes, flow, node_support, valid_mask, k=k, sigma=sigma)
    batch, edges, _ = edge_index.shape
    batch_idx = torch.arange(batch, device=nodes.device).view(batch, 1).expand(batch, edges)
    edge_valid = valid_mask[batch_idx, edge_index[..., 0]] & valid_mask[batch_idx, edge_index[..., 1]]
    c_ij, l_ij, u_ij = corridor_support(voxel_grid, nodes, edge_index, num_samples=corridor_samples)
    leaf_only = (c_ij < 0.1) & (l_ij > 0.8) & (u_ij < 0.2)
    features = build_edge_features(nodes, flow, depth, node_latents, edge_index, c_ij, l_ij, u_ij)
    edge_logits = edge_mlp(features).squeeze(-1)
    edge_valid = edge_valid & ~leaf_only
    edge_logits = edge_logits.masked_fill(~edge_valid, -20.0)
    edge_weights = soft_adjacency_from_logits(edge_logits, nodes, edge_index, c_ij, l_ij, tau=tau, sigma=sigma)
    edge_weights = edge_weights.masked_fill(~edge_valid, 0.0)
    return SparseGraphOutput(edge_index, edge_logits, edge_weights, features, edge_valid, c_ij, l_ij, u_ij)
