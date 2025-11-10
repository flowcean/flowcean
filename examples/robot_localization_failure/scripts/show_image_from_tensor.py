#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

"""
show_image_from_tensor.py

Visualizes a saved (image, label) tensor pair from the dataset cache.
Displays each individual feature channel (Map, Scan, Particles) in
grayscale and a combined RGB view to help verify correct image encoding.

Usage:
    python show_image_from_tensor.py --path learning_cache/7.pt

Example output:
    📦 File: 7.pt
    Image shape: (3, 512, 512)
    Label: 1.0 (delocalized)
"""


# --- Parse CLI ---
ap = argparse.ArgumentParser()
ap.add_argument(
    "--path",
    required=True,
    help="Path to .pt file in learning_cache/",
)
args = ap.parse_args()

path = Path(args.path)

# --- Load Tensor ---
img, label = torch.load(path, map_location="cpu")

# Extract label safely
label_value = float(label.item() if label.numel() == 1 else label[0].item())
label_str = "delocalized" if label_value >= 0.5 else "localized"

print(
    f"\n📦 File: {path.name}\n"
    f"Image shape: {tuple(img.shape)}\n"
    f"Label: {label_value:.1f} ({label_str})\n",
)

# --- Prepare figure ---
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
titles = ["Map (R)", "Scan (G)", "Particles (B)", "Combined RGB"]

# Show each grayscale channel
for i, name in enumerate(titles[:3]):
    axes[i].imshow(img[i].numpy(), cmap="gray")
    axes[i].set_title(name)
    axes[i].axis("off")

# Combined RGB (clip to [0,1] in case of float precision)
rgb = img.numpy().transpose(1, 2, 0)
rgb = rgb.clip(0, 1)
axes[3].imshow(rgb)
axes[3].set_title(titles[3])
axes[3].axis("off")

fig.suptitle(
    f"{path.name}   —   Label: {label_value:.1f} ({label_str})",
    fontsize=12,
)
plt.tight_layout()
plt.show()
