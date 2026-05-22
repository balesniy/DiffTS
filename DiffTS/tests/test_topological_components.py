import sys
from pathlib import Path

import torch
import torch.nn as nn

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from DiffTS.models.pruning_gnn import pruning_labels_from_gt
from DiffTS.utils.semantic_voxel import SemanticVoxelGrid
from DiffTS.utils.topological_batch import build_topological_batch
from DiffTS.utils.topological_decoder import decode_edmonds_arborescence
from DiffTS.utils.topological_graph import build_sparse_graph
from DiffTS.utils.topological_guidance import GuidanceConfig, guidance_displacement, guidance_eta
from DiffTS.utils.utils import compute_diffusion_params


def make_legacy_batch():
    return {
        "pcd_nodes": [torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])],
        "full_pcd_nodes": [torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])],
        "node_parent_ids": [torch.tensor([0, 0, 1])],
        "pcd_conditioning_pts": [torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])],
        "scan_point_classes": [torch.tensor([0, 1, 0])],
        "scan_point_colors": [torch.zeros(3, 3)],
        "filename": ["tiny"],
        "parent_ids": [torch.tensor([0, 0, 1])],
        "normalization_factors": [(torch.zeros(3), 1.0)],
    }


def test_topological_batch_padding_and_semantics():
    topo = build_topological_batch(make_legacy_batch(), n_max=5, device="cpu")
    assert topo.nodes_gt.shape == (1, 5, 3)
    assert topo.node_exists.tolist() == [[True, True, True, False, False]]
    assert topo.parent_ids[0, :3].tolist() == [0, 0, 1]
    assert torch.allclose(topo.scan_semantics[0][0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(topo.scan_semantics[0][1], torch.tensor([0.0, 1.0, 0.0]))


def test_x0_hat_formula_round_trip():
    diff = compute_diffusion_params(
        {
            "diff": {
                "beta_func": "linear",
                "beta_start": 2.0e-6,
                "beta_end": 2.0e-4,
                "t_steps": 10,
                "s_steps": 2,
            }
        },
        device="cpu",
    )
    x0 = torch.randn(2, 4, 3)
    eps = torch.randn_like(x0)
    t = torch.tensor([3, 7])
    sqrt_alpha = diff["sqrt_alphas_cumprod"][t][:, None, None]
    sqrt_one_minus = diff["sqrt_one_minus_alphas_cumprod"][t][:, None, None]
    x_t = sqrt_alpha * x0 + sqrt_one_minus * eps
    x0_hat = (x_t - sqrt_one_minus * eps) / sqrt_alpha
    assert torch.allclose(x0_hat, x0, atol=1e-6)


def test_semantic_voxel_grid_sample_has_coordinate_gradients():
    points = [torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])]
    semantics = [torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.2, 0.1]])]
    bounds_min = torch.tensor([[0.0, 0.0, 0.0]])
    bounds_max = torch.tensor([[1.0, 1.0, 1.0]])
    grid = SemanticVoxelGrid.from_points(points, semantics, bounds_min, bounds_max, grid_size=(5, 5, 5), blur_sigma=0.0)
    coords = torch.tensor([[[0.45, 0.45, 0.45], [0.9, 0.9, 0.9]]], requires_grad=True)
    sampled = grid.sample(coords).sum()
    sampled.backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_sparse_graph_size_and_ghost_mask():
    batch, n_nodes, k, latent_dim = 2, 8, 3, 4
    nodes = torch.rand(batch, n_nodes, 3)
    flow = torch.randn(batch, n_nodes, 3)
    depth = torch.linspace(0, 1, n_nodes).repeat(batch, 1)
    latents = torch.randn(batch, n_nodes, latent_dim)
    valid = torch.ones(batch, n_nodes, dtype=torch.bool)
    valid[:, -2:] = False
    scan_points = [torch.rand(16, 3) for _ in range(batch)]
    scan_semantics = [torch.rand(16, 3).softmax(dim=-1) for _ in range(batch)]
    bounds_min = torch.zeros(batch, 3)
    bounds_max = torch.ones(batch, 3)
    voxel = SemanticVoxelGrid.from_points(scan_points, scan_semantics, bounds_min, bounds_max, grid_size=(4, 4, 4), blur_sigma=0.0)
    edge_mlp = nn.Linear(latent_dim * 2 + 6, 1)

    graph = build_sparse_graph(nodes, flow, depth, latents, valid, voxel, edge_mlp, k=k, corridor_samples=3)

    assert graph.edge_index.shape == (batch, n_nodes * k, 2)
    assert graph.edge_index.numel() <= batch * n_nodes * k * 2
    batch_idx = torch.arange(batch).view(batch, 1).expand_as(graph.edge_valid)
    assert not valid[batch_idx, graph.edge_index[..., 1]][~graph.edge_valid].all()


def test_guidance_is_bounded_nan_safe_and_decays_to_zero():
    cfg = GuidanceConfig(active_fraction=1.0, final_decay_fraction=0.1, max_displacement=0.02, base_eta=1.0)
    x = torch.randn(1, 5, 3)

    def energy(nodes):
        return nodes.pow(2).sum()

    disp = guidance_displacement(x, energy, step_index=5, total_steps=10, cfg=cfg)
    assert torch.isfinite(disp).all()
    assert disp.norm(dim=-1).amax() <= 0.020001
    assert guidance_eta(step_index=9, total_steps=10, cfg=cfg) == 0.0


def test_edmonds_decoder_one_root_parents_and_no_ghosts():
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [10.0, 10.0, 10.0],
        ]
    )
    edge_index = torch.tensor([[0, 1], [1, 2], [3, 2], [2, 3]])
    weights = torch.tensor([2.0, 1.5, 100.0, 100.0])
    valid = torch.tensor([True, True, True, False])
    decoded = decode_edmonds_arborescence(nodes, edge_index, weights, valid_mask=valid)

    assert decoded.nodes.shape[0] == 3
    assert decoded.root_index == 0
    assert decoded.edges_index.shape[1] == 2
    parent_counts = torch.bincount(decoded.edges_index[1], minlength=3)
    assert parent_counts[decoded.root_index] == 0
    assert torch.all(parent_counts[torch.tensor([1, 2])] == 1)


def test_pruning_labels_allow_short_gt_paths():
    pred_nodes = torch.tensor([[0.0, 0, 0], [0.0, 0, 4.0], [1.0, 0, 4.0]])
    pred_edges = torch.tensor([[0, 1], [0, 2]]).t()
    gt_nodes = torch.tensor([[0.0, 0, 0], [0.0, 0, 1], [0.0, 0, 2], [0.0, 0, 3], [0.0, 0, 4]])
    gt_parent_ids = torch.tensor([0, 0, 1, 2, 3])
    labels = pruning_labels_from_gt(pred_nodes, pred_edges, gt_nodes, gt_parent_ids, max_intermediate=3)
    assert labels.tolist() == [True, True]
