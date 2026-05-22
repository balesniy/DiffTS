from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch


@dataclass
class DecodedTree:
    nodes: torch.Tensor
    edges_index: torch.Tensor
    edge_features: torch.Tensor | None
    root_index: int


def decode_edmonds_arborescence(
    nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    edge_features: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> DecodedTree:
    """Decode a maximum directed arborescence rooted at the lowest valid z node."""
    if valid_mask is None:
        valid_mask = torch.ones((nodes.shape[0],), dtype=torch.bool, device=nodes.device)
    valid_ids = torch.where(valid_mask)[0]
    if valid_ids.numel() == 0:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        return DecodedTree(nodes[:0], empty_edges, None, 0)

    valid_nodes = nodes[valid_ids]
    root_local = int(valid_nodes[:, 2].argmin().item())
    root_global = int(valid_ids[root_local].item())
    id_map = {int(global_id): local_id for local_id, global_id in enumerate(valid_ids.tolist())}

    graph = nx.DiGraph()
    super_root = "__root__"
    graph.add_node(super_root)
    for local_id in range(valid_ids.numel()):
        graph.add_node(local_id)
    root_weight = float(edge_weights.max().detach().cpu().item()) + 1.0 if edge_weights.numel() else 1.0
    graph.add_edge(super_root, root_local, weight=root_weight, feature_id=-1)

    for eid, ((parent, child), weight) in enumerate(zip(edge_index.detach().cpu().tolist(), edge_weights.detach().cpu().tolist())):
        parent = id_map.get(int(parent))
        child = id_map.get(int(child))
        if parent is None or child is None or parent == child:
            continue
        graph.add_edge(parent, child, weight=float(weight), feature_id=eid)

    if graph.number_of_edges() == 0:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        return DecodedTree(valid_nodes, empty_edges, None, root_local)

    arb = nx.maximum_spanning_arborescence(graph, attr="weight", preserve_attrs=True)
    edges = []
    feature_ids = []
    for parent, child, attrs in arb.edges(data=True):
        if parent == super_root:
            continue
        edges.append((parent, child))
        feature_ids.append(attrs.get("feature_id", -1))

    if not edges:
        edge_tensor = torch.empty((2, 0), dtype=torch.long, device=nodes.device)
        decoded_features = None
    else:
        edge_tensor = torch.tensor(edges, dtype=torch.long, device=nodes.device).t().contiguous()
        decoded_features = None
        if edge_features is not None:
            keep = [fid for fid in feature_ids if fid >= 0]
            decoded_features = edge_features[torch.tensor(keep, dtype=torch.long, device=edge_features.device)] if keep else None
    return DecodedTree(valid_nodes, edge_tensor, decoded_features, root_local)


def largest_component_mask(num_nodes: int, edges_index: torch.Tensor) -> torch.Tensor:
    if num_nodes == 0:
        return torch.zeros((0,), dtype=torch.bool, device=edges_index.device)
    if edges_index.numel() == 0:
        mask = torch.zeros((num_nodes,), dtype=torch.bool, device=edges_index.device)
        mask[0] = True
        return mask
    adjacency = [[] for _ in range(num_nodes)]
    for parent, child in edges_index.detach().cpu().t().tolist():
        adjacency[parent].append(child)
        adjacency[child].append(parent)

    visited = [False] * num_nodes
    best_component = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best_component):
            best_component = component

    mask = torch.zeros((num_nodes,), dtype=torch.bool, device=edges_index.device)
    if best_component:
        mask[torch.tensor(best_component, dtype=torch.long, device=edges_index.device)] = True
    return mask


def prune_decoded_tree(
    decoded: DecodedTree,
    keep_prob: torch.Tensor,
    threshold: float = 0.5,
) -> DecodedTree:
    if decoded.edges_index.numel() == 0:
        return decoded
    keep = keep_prob >= threshold
    edges = decoded.edges_index[:, keep]
    component = largest_component_mask(decoded.nodes.shape[0], edges)
    local_old_to_new = torch.full((decoded.nodes.shape[0],), -1, dtype=torch.long, device=decoded.nodes.device)
    local_old_to_new[component] = torch.arange(int(component.sum().item()), device=decoded.nodes.device)
    edge_keep = component[edges[0]] & component[edges[1]]
    edges = local_old_to_new[edges[:, edge_keep]]
    features = decoded.edge_features[keep][edge_keep] if decoded.edge_features is not None else None
    root = int(local_old_to_new[decoded.root_index].item()) if component[decoded.root_index] else 0
    return DecodedTree(decoded.nodes[component], edges, features, root)
