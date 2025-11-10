"""inspect_and_show_channels.py

Utility script to inspect and visualize a saved image tensor (.pt file).
It loads a (image, label) pair, prints tensor statistics such as shape,
dtype, range, and per-channel mean/std, and displays the full RGB image
along with its individual R, G, and B channels using Matplotlib.
"""

import matplotlib.pyplot as plt
import torch

path = "../learning_cache/7.pt"

# Load tensor tuple (image, label)
img, label = torch.load(path, map_location="cpu")

print(f"\nLoaded: {path}")
print(f"Image shape: {tuple(img.shape)}  dtype: {img.dtype}")
print(f"Label: {label.item()}")
print(f"Value range: {img.min().item()}  to  {img.max().item()}")

# Channel-wise stats
for i, c in enumerate(["R", "G", "B"]):
    channel = img[i]
    print(
        f"{c}-channel: mean={channel.mean().item():.5f}, "
        f"std={channel.std().item():.5f}, "
        f"min={channel.min().item():.3f}, max={channel.max().item():.3f}",
    )

# --- Visualization ---
fig, axs = plt.subplots(1, 4, figsize=(14, 4))

# Convert (C,H,W) → (H,W,C) for full RGB
axs[0].imshow(img.permute(1, 2, 0))
axs[0].set_title(f"Full RGB (label={label.item()})")

# Individual channels (gray colormap)
titles = ["Red", "Green", "Blue"]
for i in range(3):
    axs[i + 1].imshow(img[i], cmap="gray")
    axs[i + 1].set_title(titles[i])

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.show()
