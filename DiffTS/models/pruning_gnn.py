from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeUpdateLayer(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(edge_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, edge_features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        parent, child = edge_index
        agg_out = edge_features.new_zeros((num_nodes, edge_features.shape[-1]))
        agg_in = edge_features.new_zeros((num_nodes, edge_features.shape[-1]))
        agg_out.scatter_reduce_(0, parent[:, None].expand_as(edge_features), edge_features, reduce="amax", include_self=False)
        agg_in.scatter_reduce_(0, child[:, None].expand_as(edge_features), edge_features, reduce="amax", include_self=False)
        agg_out = torch.nan_to_num(agg_out, nan=0.0, neginf=0.0, posinf=0.0)
        agg_in = torch.nan_to_num(agg_in, nan=0.0, neginf=0.0, posinf=0.0)
        context = torch.cat((edge_features, agg_out[parent], agg_in[child]), dim=-1)
        return edge_features + self.update(context)


class LearnedPruningGNN(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([EdgeUpdateLayer(edge_dim, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        x = edge_features
        for layer in self.layers:
            x = layer(x, edge_index, num_nodes)
        return self.classifier(x).squeeze(-1)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


@torch.no_grad()
def pruning_labels_from_gt(
    pred_nodes: torch.Tensor,
    pred_edges: torch.Tensor,
    gt_nodes: torch.Tensor,
    gt_parent_ids: torch.Tensor,
    max_intermediate: int = 3,
) -> torch.Tensor:
    if pred_edges.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=pred_nodes.device)
    nearest_gt = torch.cdist(pred_nodes, gt_nodes).argmin(dim=-1)
    
    parent_idx = pred_edges[0]
    child_idx = pred_edges[1]
    
    gt_parent = nearest_gt[parent_idx]
    gt_child = nearest_gt[child_idx]
    
    gt_parent_ids = gt_parent_ids.long().clamp(min=0, max=max(gt_nodes.shape[0] - 1, 0))
    
    keep = (gt_child == gt_parent)
    cur = gt_child
    
    for _ in range(max_intermediate + 2):
        cur = gt_parent_ids[cur]
        keep = keep | (cur == gt_parent)
        
    return keep
