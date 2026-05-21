#!/usr/bin/env python3
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
import yaml

from DiffTS.models.models import DiffusionPoints
from DiffTS.utils.postprocess import min_spanning_tree


def merge_dicts(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_cfg(config_path: str) -> dict:
    this_dir = Path(__file__).resolve().parent
    default_cfg_path = this_dir / "config" / "default_config.yaml"

    with open(default_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    with open(config_path, "r") as f:
        exp_cfg = yaml.safe_load(f)

    merge_dicts(cfg, exp_cfg)
    return cfg


def read_ply_xyz_rgb(path: str, default_color: float = 0.5):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float32)

    if xyz.size == 0:
        raise RuntimeError(f"Empty point cloud: {path}")

    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape != xyz.shape:
        colors = np.full_like(xyz, default_color, dtype=np.float32)

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(colors).all(axis=1)
    xyz = xyz[valid]
    colors = colors[valid]

    if len(xyz) == 0:
        raise RuntimeError("No finite points after filtering NaN/Inf values.")

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    return xyz.astype(np.float32), colors


def center_like_orchard_dataset(xyz: np.ndarray, scale: float = 1.0):
    # OrchardDataset.center_and_normalize_data:
    # cloud_offset = mean(xyz), but z offset = min z
    cloud_offset = xyz.mean(axis=0).astype(np.float32)
    cloud_offset[2] = xyz[:, 2].min().astype(np.float32)

    xyz_norm = (xyz - cloud_offset) / scale
    return xyz_norm.astype(np.float32), cloud_offset.astype(np.float32), float(scale)


def sample_condition_points(xyz, colors, num_points, seed=42):
    if isinstance(num_points, str) and num_points == "auto":
        return xyz, colors

    n = int(num_points)
    rng = np.random.default_rng(seed)

    replace = len(xyz) < n
    ids = rng.choice(len(xyz), size=n, replace=replace)

    return xyz[ids].astype(np.float32), colors[ids].astype(np.float32)


def estimate_num_skeleton_nodes(xyz_norm, voxel_size, min_nodes=16, max_nodes=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_norm)
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    n = len(down.points)
    n = max(int(n), int(min_nodes))

    if max_nodes is not None:
        n = min(n, int(max_nodes))

    n = min(n, len(xyz_norm))
    return n


def make_nn_edges(nodes_xyz, parent_xyz):
    nodes_t = torch.from_numpy(nodes_xyz).float()
    parents_t = torch.from_numpy(parent_xyz).float()

    dists = torch.cdist(parents_t, nodes_t)
    eye = torch.eye(dists.shape[0], dtype=torch.bool)
    if eye.shape == dists.shape:
        dists[eye] = torch.inf

    parent_ids = dists.argmin(dim=1).cpu().numpy()

    lines = np.stack(
        [np.arange(len(nodes_xyz), dtype=np.int32), parent_ids.astype(np.int32)],
        axis=1,
    )

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(nodes_xyz.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def make_open_flow_edges(nodes_xyz, parent_xyz):
    n = len(nodes_xyz)
    pts = np.concatenate([nodes_xyz, parent_xyz], axis=0)
    lines = np.stack(
        [np.arange(n, dtype=np.int32), np.arange(n, 2 * n, dtype=np.int32)],
        axis=1,
    )

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    return ls


def save_point_cloud(path, xyz, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Input .ply point cloud")
    parser.add_argument("--weights", required=True, help="Path to DiffTS .ckpt")
    parser.add_argument(
        "--config",
        default="config/config_orchard.yaml",
        help="Path to DiffTS yaml config",
    )
    parser.add_argument("--outdir", default="outputs_ply_inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--num-points",
        default=None,
        help="Conditioning points. Default: cfg['data']['num_points']",
    )
    parser.add_argument(
        "--node-count-vox-size",
        type=float,
        default=None,
        help="Voxel size for estimating skeleton node count. Default: cfg value.",
    )
    parser.add_argument("--min-nodes", type=int, default=16)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional safety cap for predicted skeleton nodes.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Normalization scale. Orchard config usually uses 1.0.",
    )
    parser.add_argument(
        "--default-color",
        type=float,
        default=0.5,
        help="RGB value used if PLY has no colors.",
    )
    parser.add_argument(
        "--skip-mst",
        action="store_true",
        help="Skip minimum-spanning-tree postprocessing.",
    )
    parser.add_argument(
        "--max-bridge-dist",
        type=float,
        default=None,
        help="MST max_bridge_dist. Default: cfg['data']['pp_max_bridge_dist']",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DiffTS code assumes CUDA: compute_diffusion_params() and sampling use .cuda()."
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Use a CUDA device, e.g. --device cuda:0")

    torch.cuda.set_device(device.index or 0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg(args.config)

    # Important for standalone inference.
    cfg = deepcopy(cfg)
    cfg["train"]["batch_size"] = 1
    cfg["train"]["num_workers"] = 0

    if args.num_points is not None:
        cfg["data"]["num_points"] = (
            args.num_points if args.num_points == "auto" else int(args.num_points)
        )

    if args.node_count_vox_size is not None:
        cfg["data"]["node_count_vox_size"] = float(args.node_count_vox_size)

    xyz, colors = read_ply_xyz_rgb(args.ply, default_color=args.default_color)
    xyz_norm, cloud_offset, scale = center_like_orchard_dataset(xyz, scale=args.scale)

    cond_xyz_norm, cond_colors = sample_condition_points(
        xyz_norm,
        colors,
        cfg["data"]["num_points"],
        seed=args.seed,
    )

    num_skel_nodes = estimate_num_skeleton_nodes(
        cond_xyz_norm,
        voxel_size=cfg["data"]["node_count_vox_size"],
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )

    print(f"Input points: {len(xyz)}")
    print(f"Conditioning points: {len(cond_xyz_norm)}")
    print(f"Estimated skeleton nodes: {num_skel_nodes}")
    print(f"Cloud offset: {cloud_offset.tolist()}, scale: {scale}")

    print("Loading model...")
    model = DiffusionPoints.load_from_checkpoint(
        args.weights,
        hparams=cfg,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    model.freeze()

    cond_xyz_t = torch.from_numpy(cond_xyz_norm).float().to(device)
    cond_colors_t = torch.from_numpy(cond_colors).float().to(device)

    # If config uses semantic input, this gives a dummy all-zero class channel.
    cond_classes_t = torch.zeros(len(cond_xyz_norm), dtype=torch.float32, device=device)

    # diffusion_inference() uses pcd_nodes[b].shape[0] to decide how many nodes to predict.
    dummy_nodes_t = torch.zeros((num_skel_nodes, 3), dtype=torch.float32, device=device)

    batch = {
        "pcd_nodes": [dummy_nodes_t],
        "pcd_conditioning_pts": [cond_xyz_t],
        "scan_point_colors": [cond_colors_t],
        "scan_point_classes": [cond_classes_t],
    }

    print("Running diffusion inference...")
    with torch.no_grad():
        model_output = model.diffusion_inference(batch)[0].detach()

    nodes_norm_t = model_output[:, :3]
    parent_norm_t = nodes_norm_t + model_output[:, 3:]

    max_range = float(cfg["data"].get("max_range", 200.0))
    valid = nodes_norm_t.norm(dim=-1) < max_range

    if valid.sum().item() < max(1, len(valid) // 2):
        print(
            "Warning: less than half predicted nodes are inside max_range; "
            "saving valid nodes only."
        )

    nodes_norm_t = nodes_norm_t[valid]
    parent_norm_t = parent_norm_t[valid]

    nodes_norm = nodes_norm_t.cpu().numpy().astype(np.float32)
    parent_norm = parent_norm_t.cpu().numpy().astype(np.float32)

    nodes_xyz = nodes_norm * scale + cloud_offset
    parent_xyz = parent_norm * scale + cloud_offset
    cond_xyz = cond_xyz_norm * scale + cloud_offset

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ply).stem

    save_point_cloud(outdir / f"{stem}_input_sampled.ply", cond_xyz, cond_colors)
    save_point_cloud(outdir / f"{stem}_pred_nodes.ply", nodes_xyz)

    open_edges = make_open_flow_edges(nodes_xyz, parent_xyz)
    o3d.io.write_line_set(str(outdir / f"{stem}_pred_open_edges.ply"), open_edges)

    nn_edges = make_nn_edges(nodes_xyz, parent_xyz)
    o3d.io.write_line_set(str(outdir / f"{stem}_pred_nn_edges.ply"), nn_edges)

    if not args.skip_mst:
        print("Running MST postprocess...")
        max_bridge_dist = (
            float(args.max_bridge_dist)
            if args.max_bridge_dist is not None
            else float(cfg["data"].get("pp_max_bridge_dist", 0.2))
        )

        pp_edges = min_spanning_tree(
            torch.from_numpy(nodes_xyz.astype(np.float32)).to(device),
            torch.from_numpy(cond_xyz.astype(np.float32)).to(device),
            torch.from_numpy(parent_xyz.astype(np.float32)).to(device),
            max_bridge_dist=max_bridge_dist,
            connected_components_filt=bool(
                cfg["data"].get("connected_components_filt", False)
            ),
            averaging=False,
            debug=False,
        )
        o3d.io.write_line_set(str(outdir / f"{stem}_pred_mst_edges.ply"), pp_edges)

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()