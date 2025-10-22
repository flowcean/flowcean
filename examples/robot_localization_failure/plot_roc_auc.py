#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from flowcean.core.model import Model

# usage: uv run plot_roc_auc.py --model models/robot_localization.fml --cache evaluation_cache --max 5000


def load_model(model_path: str):
    wrapper = Model.load(model_path)
    model = getattr(wrapper, "module", None) or getattr(wrapper, "model", None)
    if model is None:
        raise SystemExit("❌ Could not find CNN module inside model wrapper")
    model.eval()
    return model


def get_probabilities(model, cache_dir: Path, max_samples=None):
    probs, labels = [], []
    pt_files = sorted(cache_dir.glob("*.pt"))
    if max_samples:
        pt_files = pt_files[:max_samples]

    print(f"Processing {len(pt_files)} tensors from {cache_dir} ...")

    for i, path in enumerate(pt_files, 1):
        try:
            img, label = torch.load(path, map_location="cpu")
            if not isinstance(img, torch.Tensor):
                continue

            x = img.unsqueeze(0)  # [1,3,H,W]
            with torch.no_grad():
                logits = model(x)
                prob = torch.sigmoid(logits).item()

            val = float(
                label.item() if label.numel() == 1 else label[0].item(),
            )
            probs.append(prob)
            labels.append(val)

            if i % 500 == 0 or i == len(pt_files):
                print(f"  → {i}/{len(pt_files)} processed")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

    return np.array(labels), np.array(probs)


def plot_roc(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="blue",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Delocalization Classifier")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print thresholds summary
    print("\nSample thresholds:")
    for i in np.linspace(0, len(thresholds) - 1, 5, dtype=int):
        print(
            f"  threshold={thresholds[i]:.3f}  →  TPR={tpr[i]:.3f}, FPR={fpr[i]:.3f}",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .fml model file")
    ap.add_argument("--cache", required=True, help="Folder with .pt tensors")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    args = ap.parse_args()

    model = load_model(args.model)
    labels, probs = get_probabilities(
        model,
        Path(args.cache),
        max_samples=args.max,
    )
    print(f"\nCollected {len(labels)} samples")

    plot_roc(labels, probs)


if __name__ == "__main__":
    main()
