import re
import subprocess

import torch


def print_header(title):
    print(f"\n== {title} ==")


def nvcc_release():
    result = subprocess.run(["nvcc", "--version"], check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", result.stdout)
    return match.group(1) if match else None


def check_torch_cuda(nvcc_version):
    print_header("torch")
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to torch")
    if nvcc_version and torch.version.cuda and not torch.version.cuda.startswith(nvcc_version):
        raise RuntimeError(f"torch CUDA {torch.version.cuda} does not match nvcc {nvcc_version}")
    x = torch.randn(16, 16, device="cuda")
    y = x @ x.t()
    torch.cuda.synchronize()
    print("tiny tensor op:", float(y.mean().detach().cpu()))


def minkowski_compiled_cuda_version(me_module):
    candidates = [
        getattr(me_module, "cuda_version", None),
        getattr(me_module, "get_cuda_version", None),
    ]
    try:
        import MinkowskiEngineBackend._C as backend

        candidates.extend(
            [
                getattr(backend, "cuda_version", None),
                getattr(backend, "get_cuda_version", None),
            ]
        )
    except Exception:
        pass
    for candidate in candidates:
        if callable(candidate):
            try:
                return str(candidate())
            except Exception:
                continue
        if candidate:
            return str(candidate)
    return None


def check_minkowski_engine(nvcc_version):
    print_header("MinkowskiEngine")
    import MinkowskiEngine as ME

    print("MinkowskiEngine:", getattr(ME, "__version__", "unknown"))
    compiled_cuda = minkowski_compiled_cuda_version(ME)
    print("compiled CUDA:", compiled_cuda or "not exposed")
    if compiled_cuda and nvcc_version and nvcc_version not in compiled_cuda:
        raise RuntimeError(f"MinkowskiEngine CUDA {compiled_cuda} does not match nvcc {nvcc_version}")

    coords = torch.IntTensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).cuda()
    feats = torch.randn(3, 2, device="cuda")
    sparse = ME.SparseTensor(features=feats, coordinates=coords)
    conv = ME.MinkowskiConvolution(2, 4, kernel_size=1, dimension=3).cuda()
    out = conv(sparse)
    torch.cuda.synchronize()
    print("tiny sparse conv:", tuple(out.F.shape))


def check_pykeops():
    print_header("pykeops")
    import pykeops
    from pykeops.torch import LazyTensor

    print("pykeops:", pykeops.__version__)
    x = torch.randn(8, 3, device="cuda")
    y = torch.randn(9, 3, device="cuda")
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    d_ij = ((x_i - y_j) ** 2).sum(-1)
    values = d_ij.min(dim=1)
    torch.cuda.synchronize()
    print("tiny keops op:", tuple(values.shape))


def check_open3d():
    print_header("open3d")
    import open3d as o3d

    print("open3d:", o3d.__version__)


def main():
    print_header("nvcc")
    nvcc_version = nvcc_release()
    check_torch_cuda(nvcc_version)
    check_minkowski_engine(nvcc_version)
    check_pykeops()
    check_open3d()
    print("\nCUDA stack self-check passed")


if __name__ == "__main__":
    main()
