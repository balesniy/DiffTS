from __future__ import annotations

from dataclasses import dataclass

import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning import LightningDataModule, LightningModule

import DiffTS.models.diff_skeletonizer as minknet
from DiffTS.models.diff_skeletonizer import MinkGlobalEnc, MinkUnetGlobalEnc
from DiffTS.models.pruning_gnn import LearnedPruningGNN, focal_bce_with_logits, pruning_labels_from_gt
from DiffTS.utils.semantic_voxel import SemanticVoxelGrid
from DiffTS.utils.topological_batch import TopologicalBatch, build_topological_batch
from DiffTS.utils.topological_decoder import decode_edmonds_arborescence
from DiffTS.utils.topological_graph import (
    build_sparse_graph,
    depth_topological_loss,
    two_cycle_loss,
)
from DiffTS.utils.topological_guidance import GuidanceConfig, guidance_displacement
from DiffTS.utils.utils import compute_diffusion_params, normalize_vecs, o3d_fps_sampling


@dataclass
class TopologicalModelOutput:
    eps_pred: torch.Tensor
    flow_pred: torch.Tensor
    conf_logits: torch.Tensor
    depth_pred: torch.Tensor
    node_latents: torch.Tensor
    candidate_edges: torch.Tensor | None = None
    edge_logits: torch.Tensor | None = None
    edge_features: torch.Tensor | None = None


class TopologicalDiffusionPoints(LightningModule):
    """Sparse Topological-DDPM path kept separate from legacy DiffusionPoints."""

    def __init__(self, hparams: dict, data_module: LightningDataModule | None = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.diff_params = compute_diffusion_params(self.hparams, device=self.device)
        self.model, self.partial_enc = self.init_models()
        self.dpm_scheduler = self.init_schedulers()

    def init_models(self):
        topo_cfg = self.hparams.get("topological", {})
        latent_dim = int(topo_cfg.get("latent_dim", self.hparams["model"].get("out_dim", 96)))
        semantic_channels = int(topo_cfg.get("semantic_channels", 3))
        in_channels = 3 + semantic_channels
        encoder_cls = MinkUnetGlobalEnc if self.hparams["model"].get("unet_encoder") else MinkGlobalEnc
        cond_encoder = encoder_cls(
            in_channels=in_channels,
            out_channels=latent_dim,
            downscale_model=self.hparams["model"]["downscale_params"],
            debug=self.hparams["model"]["debug"],
            multiscale_cond=self.hparams["model"]["multiscale_cond"],
        )
        denoising_model = minknet.AttentiveMinkUNetDiff(
            in_channels=6,
            out_channels=latent_dim,
            voxel_size=self.hparams["data"]["cond_resolution"],
            node_voxel_size=self.hparams["data"]["node_resolution"],
            neighb_fusion=self.hparams["train"]["neighb_fusion"],
            downscale_model=self.hparams["model"]["downscale_params"],
            debug=self.hparams["model"]["debug"],
            cond_dist_limit=self.hparams["train"]["cond_dist_limit"],
            multiscale_cond=self.hparams["model"]["multiscale_cond"],
        )
        self.eps_head = nn.Linear(latent_dim, 3)
        self.flow_head = nn.Linear(latent_dim, 3)
        self.conf_head = nn.Linear(latent_dim, 1)
        self.depth_head = nn.Linear(latent_dim, 1)
        self.node_latent_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, latent_dim))
        edge_in_dim = latent_dim * 2 + 6
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, 1))
        prune_cfg = self.hparams.get("pruning", {})
        self.pruning_gnn = LearnedPruningGNN(
            edge_in_dim + 3,
            hidden_dim=int(prune_cfg.get("hidden_dim", 128)),
            num_layers=int(prune_cfg.get("num_layers", 3)),
        )
        return denoising_model, cond_encoder

    def init_schedulers(self):
        schedulers = [
            DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams["diff"]["t_steps"],
                beta_start=self.hparams["diff"]["beta_start"],
                beta_end=self.hparams["diff"]["beta_end"],
                beta_schedule="linear",
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
            )
            for _ in range(self.hparams["train"]["batch_size"])
        ]
        for scheduler in schedulers:
            scheduler.set_timesteps(self.diff_params["s_steps"])
        return schedulers

    def torch_to_mink(self, x_feats, resolution):
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)
        x_coord = x_feats[:, :4].clone()
        x_coord[:, 1:] = torch.round(x_feats[:, 1:4] / resolution)
        return ME.TensorField(
            features=x_feats[:, 1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

    def q_sample(self, x, t, noise):
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        sqrt_alpha = self.diff_params["sqrt_alphas_cumprod"].to(x.device)[t][:, None, None]
        sqrt_one_minus = self.diff_params["sqrt_one_minus_alphas_cumprod"].to(x.device)[t][:, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.diff_params["sqrt_alphas_cumprod"].to(x_t.device)[t][:, None, None]
        sqrt_one_minus = self.diff_params["sqrt_one_minus_alphas_cumprod"].to(x_t.device)[t][:, None, None]
        return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha.clamp_min(1e-6)

    def forward_topological(self, x_full: list[torch.Tensor], topo: TopologicalBatch, t: torch.Tensor) -> TopologicalModelOutput:
        cond_input = [torch.cat((pts, sem), dim=-1) for pts, sem in zip(topo.scan_points, topo.scan_semantics)]
        part_feat = self.partial_enc(self.torch_to_mink(cond_input, resolution=self.hparams["data"]["cond_resolution"]))
        latent_list = self.model(self.torch_to_mink(x_full, resolution=self.hparams["data"]["node_resolution"]), part_feat, t)
        latents = torch.stack(latent_list, dim=0)
        node_latents = self.node_latent_head(latents)
        return TopologicalModelOutput(
            eps_pred=self.eps_head(latents),
            flow_pred=normalize_vecs(self.flow_head(latents).reshape(-1, 3)).view(latents.shape[0], latents.shape[1], 3),
            conf_logits=self.conf_head(latents).squeeze(-1),
            depth_pred=torch.sigmoid(self.depth_head(latents).squeeze(-1)),
            node_latents=node_latents,
        )

    def _parent_flow_targets(self, nodes: torch.Tensor, parent_ids: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = nodes.shape
        batch_idx = torch.arange(batch, device=nodes.device).view(batch, 1).expand(batch, n_nodes)
        parents = nodes[batch_idx, parent_ids.clamp(min=0, max=n_nodes - 1)]
        flow = normalize_vecs((parents - nodes).reshape(-1, 3)).view_as(nodes)
        flow = torch.where(exists.unsqueeze(-1), flow, torch.zeros_like(flow))
        return flow

    def _depth_targets(self, parent_ids: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
        batch, n_nodes = parent_ids.shape
        depth = torch.zeros((batch, n_nodes), dtype=torch.float32, device=parent_ids.device)
        for b in range(batch):
            for node in range(n_nodes):
                if not bool(exists[b, node]):
                    continue
                cur = node
                seen = set()
                steps = 0
                while cur not in seen:
                    seen.add(cur)
                    parent = int(parent_ids[b, cur].item())
                    if parent == cur or parent < 0:
                        break
                    steps += 1
                    cur = parent
                depth[b, node] = steps
        max_depth = depth.amax(dim=1, keepdim=True).clamp_min(1.0)
        return depth / max_depth

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        return ((pred - target).pow(2) * mask_f).sum() / (mask_f.sum() * pred.shape[-1]).clamp_min(1.0)

    def _edge_labels(self, edge_index: torch.Tensor, parent_ids: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
        batch, edges, _ = edge_index.shape
        batch_idx = torch.arange(batch, device=edge_index.device).view(batch, 1).expand(batch, edges)
        parent = edge_index[..., 0]
        child = edge_index[..., 1]
        labels = parent_ids[batch_idx, child] == parent
        labels = labels & exists[batch_idx, parent] & exists[batch_idx, child]
        return labels.float()

    def _conf_valid_mask(self, conf_logits: torch.Tensor, threshold: float) -> torch.Tensor:
        valid = torch.sigmoid(conf_logits) >= threshold
        best = conf_logits.argmax(dim=1)
        valid[torch.arange(conf_logits.shape[0], device=conf_logits.device), best] = True
        return valid

    def _guidance_config(self) -> GuidanceConfig:
        cfg = self.hparams.get("guidance", {})
        return GuidanceConfig(
            active_fraction=float(cfg.get("active_fraction", 0.2)),
            final_decay_fraction=float(cfg.get("final_decay_fraction", 0.05)),
            max_displacement=float(cfg.get("max_displacement", 0.02)),
            base_eta=float(cfg.get("base_eta", 1.0)),
            repel_weight=float(cfg.get("repel_weight", 0.0)),
            repel_radius=float(cfg.get("repel_radius", 0.005)),
            repel_k=int(cfg.get("repel_k", 8)),
        )

    def _sample_initial_nodes(self, points: torch.Tensor, n_max: int) -> torch.Tensor:
        if points.shape[0] == 0:
            return points.new_zeros((n_max, 3))
        if points.shape[0] >= n_max:
            return o3d_fps_sampling(points, n_max).to(dtype=points.dtype)
        idx = torch.randint(0, points.shape[0], (n_max,), device=points.device)
        return points[idx]

    def _pruning_features(self, graph, nodes: torch.Tensor, batch_idx: int, edge_mask: torch.Tensor) -> torch.Tensor:
        features = graph.edge_features[batch_idx, edge_mask]
        edge_index = graph.edge_index[batch_idx, edge_mask]
        if features.numel() == 0:
            return features.new_zeros((0, features.shape[-1] + 3))
        parent = edge_index[:, 0]
        child = edge_index[:, 1]
        length = (nodes[batch_idx, child] - nodes[batch_idx, parent]).norm(dim=-1, keepdim=True)
        rel_length = length / length.mean().clamp_min(1e-6)
        out_degree = torch.bincount(parent, minlength=nodes.shape[1]).float().to(nodes.device)
        parent_out_degree = out_degree[parent].unsqueeze(-1) / max(1, int(edge_mask.sum().item()))
        edge_logit = graph.edge_logits[batch_idx, edge_mask].unsqueeze(-1)
        return torch.cat((features, rel_length, parent_out_degree, edge_logit), dim=-1)

    def _pruning_loss(self, graph, nodes: torch.Tensor, topo: TopologicalBatch) -> torch.Tensor:
        prune_cfg = self.hparams.get("pruning", {})
        terms = []
        for b in range(nodes.shape[0]):
            edge_mask = graph.edge_valid[b]
            if not edge_mask.any():
                continue
            edge_index = graph.edge_index[b, edge_mask].t().contiguous()
            features = self._pruning_features(graph, nodes, b, edge_mask)
            logits = self.pruning_gnn(features, edge_index, nodes.shape[1])
            labels = pruning_labels_from_gt(
                nodes[b].detach(),
                edge_index,
                topo.highres_gt_nodes[b],
                topo.highres_gt_edges[b],
                max_intermediate=int(prune_cfg.get("max_intermediate", 3)),
            )
            terms.append(
                focal_bce_with_logits(
                    logits,
                    labels.float(),
                    alpha=float(prune_cfg.get("focal_alpha", 0.25)),
                    gamma=float(prune_cfg.get("focal_gamma", 2.0)),
                )
            )
        return torch.stack(terms).mean() if terms else nodes.new_zeros(())

    def _pruned_edge_mask(self, graph, nodes: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prune_cfg = self.hparams.get("pruning", {})
        edge_mask = graph.edge_valid[batch_idx].clone()
        keep_logits = graph.edge_logits.new_empty((0,))
        if not prune_cfg.get("enabled", True) or not edge_mask.any():
            return edge_mask, keep_logits
        edge_ids = torch.where(edge_mask)[0]
        edge_index = graph.edge_index[batch_idx, edge_mask].t().contiguous()
        features = self._pruning_features(graph, nodes, batch_idx, edge_mask)
        keep_logits = self.pruning_gnn(features, edge_index, nodes.shape[1])
        keep = torch.sigmoid(keep_logits) >= float(prune_cfg.get("threshold", 0.5))
        edge_mask[edge_ids] = keep
        return edge_mask, keep_logits

    def training_step(self, batch: dict, batch_idx: int):
        topo_cfg = self.hparams.get("topological", {})
        loss_cfg = self.hparams.get("losses", {})
        graph_cfg = self.hparams.get("graph", {})
        voxel_cfg = self.hparams.get("voxel", {})
        n_max = int(topo_cfg.get("n_max", self.hparams["data"].get("num_nodes", 3000)))
        if isinstance(self.hparams["data"].get("num_nodes"), str):
            n_max = int(topo_cfg.get("n_max", 3000))
        topo = build_topological_batch(batch, n_max=n_max, device=self.device, bounds_padding=float(voxel_cfg.get("bounds_padding", 0.05)))

        batch_size = topo.nodes_gt.shape[0]
        t = torch.randint(0, self.diff_params["t_steps"], size=(batch_size,), device=self.device).long()
        node_noise = torch.randn_like(topo.nodes_gt)
        flow_target = self._parent_flow_targets(topo.nodes_gt, topo.parent_ids, topo.node_exists)
        flow_noise = torch.randn_like(flow_target)
        node_t = self.q_sample(topo.nodes_gt, t, node_noise)
        flow_t = flow_target + self.q_sample(torch.zeros_like(flow_target), t, flow_noise)
        x_full = [torch.cat((node_t[b], flow_t[b]), dim=-1) for b in range(batch_size)]

        out = self.forward_topological(x_full, topo, t)
        x0_hat = self.predict_x0_from_eps(node_t, t, out.eps_pred)
        depth_target = self._depth_targets(topo.parent_ids, topo.node_exists)

        loss_eps = self._masked_mse(out.eps_pred, node_noise, topo.node_exists)
        loss_flow = self._masked_mse(out.flow_pred, flow_target, topo.node_exists)
        pos_weight = (topo.node_exists.numel() - topo.node_exists.sum()).float() / topo.node_exists.sum().float().clamp_min(1.0)
        loss_conf = F.binary_cross_entropy_with_logits(out.conf_logits, topo.node_exists.float(), pos_weight=pos_weight.clamp_min(1.0))
        loss_depth = F.mse_loss(out.depth_pred[topo.node_exists], depth_target[topo.node_exists]) if topo.node_exists.any() else out.depth_pred.new_zeros(())

        voxel_grid = SemanticVoxelGrid.from_points(
            topo.scan_points,
            topo.scan_semantics,
            topo.bounds_min,
            topo.bounds_max,
            grid_size=voxel_cfg.get("grid_size", (128, 128, 256)),
            blur_sigma=float(voxel_cfg.get("blur_sigma", 1.0)),
        )
        graph = build_sparse_graph(
            x0_hat,
            out.flow_pred,
            out.depth_pred,
            out.node_latents,
            topo.node_exists,
            voxel_grid,
            self.edge_mlp,
            k=int(graph_cfg.get("k", 12)),
            corridor_samples=int(graph_cfg.get("corridor_samples", 8)),
            sigma=float(graph_cfg.get("sigma", 0.1)),
            tau=float(graph_cfg.get("tau", 0.25)),
        )
        out.candidate_edges = graph.edge_index
        out.edge_logits = graph.edge_logits
        out.edge_features = graph.edge_features
        edge_labels = self._edge_labels(graph.edge_index, topo.parent_ids, topo.node_exists)
        if graph.edge_valid.any():
            loss_edge = F.binary_cross_entropy_with_logits(graph.edge_logits[graph.edge_valid], edge_labels[graph.edge_valid])
        else:
            loss_edge = graph.edge_logits.new_zeros(())
        loss_top = depth_topological_loss(graph.edge_index, graph.edge_weights, out.depth_pred, margin=float(graph_cfg.get("depth_margin", 0.05)))
        if graph_cfg.get("use_two_cycle_loss", False):
            loss_top = loss_top + two_cycle_loss(graph.edge_index, graph.edge_weights)
        loss_prune = self._pruning_loss(graph, x0_hat, topo) if self.hparams.get("pruning", {}).get("enabled", True) else x0_hat.new_zeros(())

        total = (
            loss_cfg.get("eps", 1.0) * loss_eps
            + loss_cfg.get("flow", 1.0) * loss_flow
            + loss_cfg.get("conf", 1.0) * loss_conf
            + loss_cfg.get("depth", 0.2) * loss_depth
            + loss_cfg.get("edge", 1.0) * loss_edge
            + loss_cfg.get("topology", 0.1) * loss_top
            + loss_cfg.get("pruning", 0.25) * loss_prune
        )
        self.log_dict(
            {
                "train/loss": total,
                "train/loss_eps": loss_eps,
                "train/loss_flow": loss_flow,
                "train/loss_conf": loss_conf,
                "train/loss_depth": loss_depth,
                "train/loss_edge": loss_edge,
                "train/loss_topology": loss_top,
                "train/loss_pruning": loss_prune,
            },
            on_step=True,
            prog_bar=False,
        )
        return total

    def diffusion_inference(self, batch: dict):
        topo_cfg = self.hparams.get("topological", {})
        graph_cfg = self.hparams.get("graph", {})
        voxel_cfg = self.hparams.get("voxel", {})
        n_max = int(topo_cfg.get("n_max", self.hparams["data"].get("num_nodes", 3000) if isinstance(self.hparams["data"].get("num_nodes"), int) else 3000))
        topo = build_topological_batch(batch, n_max=n_max, device=self.device, bounds_padding=float(voxel_cfg.get("bounds_padding", 0.05)))
        init_pts = [self._sample_initial_nodes(topo.scan_points[b], n_max) for b in range(len(topo.scan_points))]
        x = torch.stack(init_pts, dim=0)
        flow = torch.zeros_like(x)
        voxel_grid = SemanticVoxelGrid.from_points(
            topo.scan_points,
            topo.scan_semantics,
            topo.bounds_min,
            topo.bounds_max,
            grid_size=voxel_cfg.get("grid_size", (128, 128, 256)),
            blur_sigma=float(voxel_cfg.get("blur_sigma", 1.0)),
        )
        guidance_cfg = self._guidance_config()
        out = None
        for scheduler in self.dpm_scheduler:
            scheduler.set_timesteps(self.diff_params["s_steps"])
        timesteps = self.dpm_scheduler[0].timesteps
        total_steps = len(timesteps)
        final_graph = None
        valid_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        for step_idx, timestep in enumerate(timesteps):
            t = torch.full((x.shape[0],), int(timestep), dtype=torch.long, device=self.device)
            full = [torch.cat((x[b], flow[b]), dim=-1) for b in range(x.shape[0])]
            out = self.forward_topological(full, topo, t)
            x0_hat = self.predict_x0_from_eps(x, t, out.eps_pred)
            valid_mask = self._conf_valid_mask(out.conf_logits, threshold=float(graph_cfg.get("conf_threshold", 0.5)))

            def energy_fn(candidate_nodes: torch.Tensor) -> torch.Tensor:
                graph = build_sparse_graph(
                    candidate_nodes,
                    out.flow_pred.detach(),
                    out.depth_pred.detach(),
                    out.node_latents.detach(),
                    valid_mask,
                    voxel_grid,
                    self.edge_mlp,
                    k=int(graph_cfg.get("k", 12)),
                    corridor_samples=int(graph_cfg.get("corridor_samples", 8)),
                    sigma=float(graph_cfg.get("sigma", 0.1)),
                    tau=float(graph_cfg.get("tau", 0.25)),
                )
                edge_term = -graph.edge_weights[graph.edge_valid].mean() if graph.edge_valid.any() else candidate_nodes.new_zeros(())
                topo_term = depth_topological_loss(
                    graph.edge_index,
                    graph.edge_weights,
                    out.depth_pred.detach(),
                    margin=float(graph_cfg.get("depth_margin", 0.05)),
                )
                return edge_term + topo_term

            x0_hat = x0_hat + guidance_displacement(
                x0_hat,
                energy_fn,
                step_idx,
                total_steps,
                guidance_cfg,
                valid_mask=valid_mask,
            )
            x = x0_hat
            flow = out.flow_pred
        assert out is not None
        final_graph = build_sparse_graph(
            x,
            flow,
            out.depth_pred,
            out.node_latents,
            valid_mask,
            voxel_grid,
            self.edge_mlp,
            k=int(graph_cfg.get("k", 12)),
            corridor_samples=int(graph_cfg.get("corridor_samples", 8)),
            sigma=float(graph_cfg.get("sigma", 0.1)),
            tau=float(graph_cfg.get("tau", 0.25)),
        )
        out.candidate_edges = final_graph.edge_index
        out.edge_logits = final_graph.edge_logits
        out.edge_features = final_graph.edge_features
        decoded = []
        pruning_logits = []
        for b in range(x.shape[0]):
            edge_mask, keep_logits = self._pruned_edge_mask(final_graph, x, b)
            pruning_logits.append(keep_logits)
            decoded.append(
                decode_edmonds_arborescence(
                    x[b],
                    final_graph.edge_index[b, edge_mask],
                    final_graph.edge_weights[b, edge_mask],
                    final_graph.edge_features[b, edge_mask],
                    valid_mask[b],
                )
            )
        return {
            "nodes": x,
            "flow": flow,
            "conf_logits": out.conf_logits,
            "depth": out.depth_pred,
            "node_latents": out.node_latents,
            "candidate_edges": final_graph.edge_index,
            "edge_logits": final_graph.edge_logits,
            "edge_features": final_graph.edge_features,
            "edge_valid": final_graph.edge_valid,
            "pruning_logits": pruning_logits,
            "decoded": decoded,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["train"]["lr"], betas=(0.9, 0.999))
