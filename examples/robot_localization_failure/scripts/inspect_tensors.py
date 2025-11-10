"""inspect_tensors.py

General-purpose inspection tool for PyTorch .pt files.
Loads the specified file, detects its structure (tensor, list, dict, etc.),
and prints summaries including type, shape, dtype, range, mean/std, and
sample values. If the tensor looks like an image (HxW or 3xHxW), a visual
preview is displayed using Matplotlib.
Usage:
    python inspect_tensors.py <path/to/file.pt>
"""

import sys
from pathlib import Path

import torch


def summarize_tensor(t: torch.Tensor, name="tensor"):
    t_cpu = t.detach().to("cpu")
    print(f"\n[{name}]")
    print(f"  type:     {type(t)}")
    print(f"  dtype:    {t_cpu.dtype}")
    print(f"  shape:    {tuple(t_cpu.shape)}")
    print(f"  min/max:  {t_cpu.min().item():.6g} / {t_cpu.max().item():.6g}")
    mean = t_cpu.float().mean().item() if t_cpu.numel() > 0 else float("nan")
    std = (
        t_cpu.float().std(unbiased=False).item()
        if t_cpu.numel() > 1
        else float("nan")
    )
    print(f"  mean/std: {mean:.6g} / {std:.6g}")

    # Small sample of values (first few elements flattened)
    flat = t_cpu.flatten()
    sample = flat[: min(10, flat.numel())].tolist()
    print(f"  sample:   {sample}")


def try_show_image(obj):
    """Best-effort image preview if it's HxW, 1xHxW, 3xHxW, or HxWx3."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return  # matplotlib not available

    if not isinstance(obj, torch.Tensor):
        return
    t = obj.detach().to("cpu")
    if t.ndim == 2:  # H, W
        img = t.numpy()
    elif t.ndim == 3:
        # 1xHxW or 3xHxW or HxWx3
        if t.shape[0] in (1, 3):
            img = t.numpy().transpose(1, 2, 0)
        elif t.shape[-1] in (1, 3):
            img = t.numpy()
        else:
            return
    else:
        return

    # Normalize to [0,1] for display if needed
    img_np = img.astype("float32") if isinstance(img, (list, tuple)) else img
    img_np = img_np - img_np.min() if img_np.max() > img_np.min() else img_np
    img_np = img_np / img_np.max() if img_np.max() > 0 else img_np

    plt.figure()
    if img_np.ndim == 2 or img_np.shape[-1] == 1:
        plt.imshow(img_np.squeeze(), cmap="gray")
    else:
        plt.imshow(img_np)
    plt.title("Preview")
    plt.axis("off")
    plt.show()


def summarize_object(obj, name="object"):
    if isinstance(obj, torch.Tensor):
        summarize_tensor(obj, name)
        try_show_image(obj)
    elif isinstance(obj, dict):
        print(f"\n[{name}] dict with keys: {list(obj.keys())}")
        # Show quick summaries for tensor-like values
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                summarize_tensor(v, f"{name}.{k}")
            else:
                t = type(v).__name__
                short = str(v)
                if len(short) > 120:
                    short = short[:117] + "..."
                print(f"  {k}: ({t}) {short}")
    elif isinstance(obj, (list, tuple)):
        print(f"\n[{name}] {type(obj).__name__} of length {len(obj)}")
        for i, v in enumerate(obj[:5]):  # limit to first few
            if isinstance(v, torch.Tensor):
                summarize_tensor(v, f"{name}[{i}]")
            else:
                t = type(v).__name__
                short = str(v)
                if len(short) > 120:
                    short = short[:117] + "..."
                print(f"  [{i}]: ({t}) {short}")
        if len(obj) > 5:
            print(f"  ... ({len(obj) - 5} more items)")
    else:
        print(f"\n[{name}] {type(obj)}")
        s = str(obj)
        print(s if len(s) < 2000 else s[:1997] + "...")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_tensors.py <path/to/file.pt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading: {path}")
    obj = torch.load(path, map_location="cpu")
    summarize_object(obj, name=path.name)


if __name__ == "__main__":
    main()
