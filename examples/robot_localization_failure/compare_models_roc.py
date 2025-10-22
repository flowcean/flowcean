#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from flowcean.core.model import Model

# ============================================================
# 🧩 Configuration — specify your models here
# ============================================================
MODEL_PATHS = [
    "models/robot_localization_1_0_10epochs.fml",
    "models/robot_localization_2_5_10epochs.fml",
    "models/robot_localization_2_5_20epochs.fml",
    "models/robot_localization_1_0_40epochs.fml",
]

CACHE_DIR = Path("evaluation_cache")
MAX_SAMPLES = 10000  # optional: limit to speed up evaluation


# ============================================================
# Helper: Load model and extract CNN
# ============================================================
def load_model(model_path: str):
    wrapper = Model.load(model_path)
    model = getattr(wrapper, "module", None) or getattr(wrapper, "model", None)
    if model is None:
        raise SystemExit(
            f"❌ Could not find CNN inside wrapper for {model_path}",
        )
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

    for path in pt_files:
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
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

    return np.array(labels), np.array(probs)


# ============================================================
# Main: Compare ROC curves
# ============================================================
def main():
    plt.figure(figsize=(8, 7))

    for model_path in MODEL_PATHS:
        model_name = Path(model_path).stem
        print(f"\n🚀 Evaluating model: {model_name}")

        model = load_model(model_path)
        labels, probs = get_probabilities(
            model,
            CACHE_DIR,
            max_samples=MAX_SAMPLES,
        )
        print(f"  → Collected {len(labels)} samples")

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison of Multiple Models")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
