#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from flowcean.core.model import Model


# ============================================================
# Helper: Load model and extract CNN
# ============================================================
def load_model(model_path: str):
    wrapper = Model.load(model_path)
    model = getattr(wrapper, "module", None) or getattr(wrapper, "model", None)
    if model is None:
        raise SystemExit("❌ Could not find CNN module inside model wrapper")
    model.eval()
    return model


# ============================================================
# Helper: Iterate through cached tensors and get probabilities
# ============================================================
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
                label.item() if label.numel() == 1 else label[0].item()
            )
            probs.append(prob)
            labels.append(val)

            if i % 500 == 0 or i == len(pt_files):
                print(f"  → {i}/{len(pt_files)} processed")
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

    return np.array(labels), np.array(probs)


# ============================================================
# Helper: Evaluate metrics at a given threshold
# ============================================================
def evaluate_threshold(labels, probs, threshold, name=""):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    print(f"\n📊 Metrics {name} (threshold={threshold:.3f})")
    print("Confusion Matrix:")
    print(cm)
    print(
        f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}",
    )


# ============================================================
# Main plotting + evaluation
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .fml model file")
    ap.add_argument("--cache", required=True, help="Folder with .pt tensors")
    ap.add_argument(
        "--max", type=int, default=None, help="Limit number of samples"
    )
    args = ap.parse_args()

    # Load model and data
    model = load_model(args.model)
    labels, probs = get_probabilities(
        model, Path(args.cache), max_samples=args.max
    )
    print(f"\nCollected {len(labels)} samples")

    # ======================================================
    # ROC + AUC computation
    # ======================================================
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Densify curve (optional)
    fpr_dense = np.linspace(0, 1, 2000)
    tpr_dense = np.interp(fpr_dense, fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr_dense, tpr_dense, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}"
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Delocalization Classifier")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ======================================================
    # Optimal threshold
    # ======================================================
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\n✅ Optimal threshold (max TPR - FPR): {optimal_threshold:.4f}")
    print(f"   TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}")

    # ======================================================
    # Evaluate metrics
    # ======================================================
    evaluate_threshold(labels, probs, 0.5, name="at 0.5")
    evaluate_threshold(
        labels, probs, optimal_threshold, name="at optimal threshold"
    )


if __name__ == "__main__":
    main()
