#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numpy(x, name):
    if x is None:
        raise KeyError(f"Missing required key: {name}")

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    return x


def to_xyz(x, name):
    x = to_numpy(x, name).astype(np.float32)

    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N, 3], got {x.shape}")

    return x[:, :3]


def to_edges(x, name):
    x = to_numpy(x, name).astype(np.int64)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"{name} must have shape [E, 2], got {x.shape}")

    return x


def colors_to_uint8(colors, n, default=(160, 160, 160)):
    if colors is None:
        return np.tile(np.array(default, dtype=np.uint8), (n, 1))

    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    colors = np.asarray(colors)

    if colors.ndim != 2 or colors.shape[0] != n or colors.shape[1] < 3:
        return np.tile(np.array(default, dtype=np.uint8), (n, 1))

    colors = colors[:, :3]

    if colors.dtype != np.uint8:
        if np.nanmax(colors) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    return colors


def filter_finite_points(points, colors=None):
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    if colors is not None:
        colors = colors[mask]

    return points, colors


def filter_valid_edges(edges, num_vertices):
    valid = (
            (edges[:, 0] >= 0)
            & (edges[:, 0] < num_vertices)
            & (edges[:, 1] >= 0)
            & (edges[:, 1] < num_vertices)
            & (edges[:, 0] != edges[:, 1])
    )
    return edges[valid]


def write_ply_vertices(path, vertices, colors):
    path = Path(path)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(vertices, colors):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def write_ply_graph(path, vertices, edges, vertex_color=(255, 40, 40), edge_color=(255, 0, 0)):
    path = Path(path)

    vertex_colors = np.tile(np.array(vertex_color, dtype=np.uint8), (len(vertices), 1))
    edge_colors = np.tile(np.array(edge_color, dtype=np.uint8), (len(edges), 1))

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        for p, c in zip(vertices, vertex_colors):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

        for e, c in zip(edges, edge_colors):
            f.write(
                f"{int(e[0])} {int(e[1])} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def write_ply_combined(
        path,
        cloud_points,
        cloud_colors,
        skel_vertices,
        skel_edges,
        skel_vertex_color=(255, 40, 40),
        edge_color=(255, 0, 0),
):
    path = Path(path)

    skel_colors = np.tile(
        np.array(skel_vertex_color, dtype=np.uint8),
        (len(skel_vertices), 1),
    )

    all_vertices = np.concatenate([cloud_points, skel_vertices], axis=0)
    all_colors = np.concatenate([cloud_colors, skel_colors], axis=0)

    edge_offset = len(cloud_points)
    combined_edges = skel_edges + edge_offset

    edge_colors = np.tile(
        np.array(edge_color, dtype=np.uint8),
        (len(combined_edges), 1),
    )

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {len(all_vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write(f"element edge {len(combined_edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        for p, c in zip(all_vertices, all_colors):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

        for e, c in zip(combined_edges, edge_colors):
            f.write(
                f"{int(e[0])} {int(e[1])} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def parse_rgb(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be R,G,B")

    rgb = tuple(int(x) for x in parts)

    if any(v < 0 or v > 255 for v in rgb):
        raise argparse.ArgumentTypeError("Color values must be in [0, 255]")

    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="DiffTS dataset .pt file")
    parser.add_argument("--outdir", "-o", default="ply_from_dataset")
    parser.add_argument("--prefix", default=None)

    parser.add_argument(
        "--cloud-color",
        type=parse_rgb,
        default="160,160,160",
        help="Fallback cloud color if scan_point_colors is absent. Format: R,G,B",
    )
    parser.add_argument(
        "--skeleton-color",
        type=parse_rgb,
        default="255,40,40",
        help="Skeleton vertex color. Format: R,G,B",
    )
    parser.add_argument(
        "--edge-color",
        type=parse_rgb,
        default="255,0,0",
        help="Skeleton edge color. Format: R,G,B",
    )

    parser.add_argument(
        "--normalize-like-orchard",
        action="store_true",
        help=(
            "Apply DiffTS OrchardDataset-style normalization: "
            "offset = mean XYZ, but offset.z = min Z."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or input_path.stem

    data = torch_load(input_path)

    scan_points = to_xyz(data.get("scan_points"), "scan_points")
    skeleton_vertices = to_xyz(data.get("skeleton_vertices"), "skeleton_vertices")
    skeleton_edges = to_edges(data.get("skeleton_edges"), "skeleton_edges")

    scan_colors = colors_to_uint8(
        data.get("scan_point_colors"),
        len(scan_points),
        default=args.cloud_color,
    )

    scan_points, scan_colors = filter_finite_points(scan_points, scan_colors)
    skeleton_vertices, _ = filter_finite_points(skeleton_vertices, None)
    skeleton_edges = filter_valid_edges(skeleton_edges, len(skeleton_vertices))

    if args.normalize_like_orchard:
        cloud_offset = scan_points.mean(axis=0)
        cloud_offset[2] = scan_points[:, 2].min()

        scan_points = scan_points - cloud_offset
        skeleton_vertices = skeleton_vertices - cloud_offset

        print(f"Applied Orchard-style offset: {cloud_offset.tolist()}")

    cloud_path = outdir / f"{prefix}_cloud.ply"
    graph_path = outdir / f"{prefix}_gt_graph.ply"
    combined_path = outdir / f"{prefix}_combined_cloud_gt.ply"

    write_ply_vertices(cloud_path, scan_points, scan_colors)

    write_ply_graph(
        graph_path,
        skeleton_vertices,
        skeleton_edges,
        vertex_color=args.skeleton_color,
        edge_color=args.edge_color,
    )

    write_ply_combined(
        combined_path,
        scan_points,
        scan_colors,
        skeleton_vertices,
        skeleton_edges,
        skel_vertex_color=args.skeleton_color,
        edge_color=args.edge_color,
    )

    print("Done.")
    print(f"Input file:          {input_path}")
    print(f"Scan points:         {len(scan_points)}")
    print(f"GT skeleton nodes:   {len(skeleton_vertices)}")
    print(f"GT skeleton edges:   {len(skeleton_edges)}")
    print()
    print(f"Cloud PLY:           {cloud_path}")
    print(f"GT graph PLY:        {graph_path}")
    print(f"Combined PLY:        {combined_path}")


if __name__ == "__main__":
    main()